from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

from models.activations import make_activation
from models.attention import ProximityAttention, apply_influence_scores, attention_weights
from models.checkpoint import REQUIRED_ON_SAVE, load_into_model, save_checkpoint
from models.features import Encoding, as_tcnn_config
from models.init import initial_points
from models.regularizers import scheduled_loss, scheduled_weight
from models.sh import SH_C0
from models.topk import TopkSelection, fused_projection, select_topk_points
from models.unet import unet_from_config
from utils.runtime import to_config_node

__all__ = [
    "PAPR",
    "EvalOutput",
    "PointCloud",
    "RenderOutput",
    "rgb_to_sh",
]

#: Parameters whose leading dimension is the point count. :mod:`models.densify` replaces all of
#: them together when it prunes or adds; :mod:`models.checkpoint` resizes them on load.
PER_POINT_PARAMETERS: tuple[str, ...] = (
    "points",
    "pc_feats",
    "points_influ_scores",
    "points_scaler",
    "points_normals",
    "points_last_grad",
    "points_acc_grad",
    "points_acc_grad_norm",
    "points_grad_cnt",
    "points_density",
)

_TOPK_DTYPES: Mapping[str, torch.dtype] = {"float32": torch.float32, "float16": torch.float16}


def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Colour to a degree-0 spherical-harmonic coefficient ."""
    return (rgb - 0.5) / SH_C0


class PointCloud(NamedTuple):
    """The point tensor a single render actually scored, and the bookkeeping that goes with it.

    ``points`` is the *augmented* tensor: real points first, then the background point cloud when
    ``use_bkg_points`` is on, then subsetted by per-step dropout. ``select_k_ind`` from a render
    indexes **this** tensor, not ``model.points`` -- which is precisely why the two travel together
    instead of the indices being recomputed against whatever ``self.points`` happens to hold.
    """

    #: ``(M, 3)`` the points this render scored.
    points: Tensor
    #: ``(M, 1)`` 1.0 for a real point, 0.0 for a background-sphere point.
    bkg_points_mask: Tensor
    #: Indices into the *un-dropped* augmented cloud, or ``None`` when no dropout was applied.
    #: Per-point tensors (features, influence, scaler) must be gathered with this before use.
    kept_indices: Tensor | None


class RenderOutput(NamedTuple):
    """Everything :meth:`PAPR.forward` produced, including what the regularizers need.

    ``int_rgb`` and ``hit_pred`` are always ``None`` and are dropped; ``grid`` came from ray
    perturbation, which this release refuses at construction. The regularizer dict moves to
    :mod:`models.regularizers`, which reads it off this bundle.
    """

    #: ``(N, H, W, 3)`` predicted colour, before ``models.last_act``.
    rgb: Tensor
    #: ``(N, H, W, C)`` attention-blended value features, the U-Net's input.
    fused_features: Tensor
    #: ``(N, H, W, K+1, 1)`` attention weights, background slot last.
    attn: Tensor
    #: ``(N, H, W, K, 1)`` the foreground slice of ``attn``.
    topk_attn: Tensor
    #: ``(N, H, W, 1)`` the background slot's weight. Note the rank: indexing with
    #: ``attn[..., -1, :]`` (an index, not a slice) means this has *no* slot axis.
    bkg_attn: Tensor
    #: ``(N, H, W, K)`` indices into :attr:`PointCloud.points`.
    select_k_ind: Tensor
    #: The cloud those indices address.
    cloud: PointCloud
    #: ``(N, H, W)`` pixel distance to the nearest projected point. Computed by the ``d2r_z``
    #: selector; zeros on the ``knn_frustum`` path, which never forms the distance matrix it would
    #: come from. Nothing in this release consumes it -- see :func:`models.topk.select_topk_points`.
    min_d2r: Tensor
    #: ``(N, H, W, 1, 3)`` where each ray leaves the background sphere.
    sphere_intersection: Tensor
    #: ``(N, H, W, K+1, 1)`` along-ray depth of every slot.
    pd: Tensor
    #: ``(N, H, W, K+1, 1)`` perpendicular ray distance of every slot.
    d2r: Tensor
    #: ``(N, H, W, K+1, C)`` per-slot value features, before blending.
    values: Tensor


class EvalOutput(NamedTuple):
    """What :meth:`PAPR.evaluate` produced for one tile.

    ``fused_features`` keeps the singleton slot axis because the caller accumulates tiles into an
    ``(N, H, W, 1, C)`` frame buffer and decodes the whole frame once. Decoding per tile would put
    a U-Net receptive field boundary at every tile seam.
    """

    #: ``(N, H, W, 1, C)`` attention-blended value features.
    fused_features: Tensor
    #: ``(N, H, W, K+1, 1)`` attention weights, background slot last.
    attn: Tensor
    #: ``(N, H, W, K)`` indices into :attr:`PointCloud.points`. Returned, never stashed.
    select_k_ind: Tensor
    #: The cloud those indices address.
    cloud: PointCloud
    #: ``(N, H, W)`` pixel distance to the nearest projected point. Computed by the ``d2r_z``
    #: selector; zeros on the ``knn_frustum`` path, which never forms the distance matrix it would
    #: come from. Nothing in this release consumes it -- see :func:`models.topk.select_topk_points`.
    min_d2r: Tensor
    #: ``(N, H, W, 1, 3)`` background-sphere exit point.
    sphere_intersection: Tensor
    #: ``(N, H, W, K+1, 1)`` along-ray depth of every slot.
    pd: Tensor
    #: ``(N, H, W, K+1, 1)`` perpendicular ray distance of every slot.
    d2r: Tensor
    #: ``(N, H, W, K+1, C)`` per-slot value features, before blending.
    values: Tensor


class PAPR(nn.Module):
    """Proximity Attention Point Rendering.

    Args:
        args: The resolved config tree (``utils.config.load_config(...).values``). The whole tree,
            not a subset -- the model reads ``geoms``, ``models``, ``training`` and several
            top-level keys.
        device: Where the *side* tensors are allocated during construction. Note that this is
            deliberately not uniform: ``points``, ``pc_feats``, ``bkg_feats``,
            ``append_bkg_points_feats`` and the U-Net are built on the CPU, because that is which
            RNG stream their initial values come from. The caller moves the finished model with
            ``.to(device)``.

    The camera intrinsics are not config: they come from the dataset and must be stamped on with
    :meth:`set_camera` before the first render.
    """

    #: See the module-level constant. Exposed on the class so :mod:`models.densify` does not have
    #: to import a private name.
    PER_POINT_PARAMETERS = PER_POINT_PARAMETERS

    def __init__(self, args: Any, device: str | torch.device = "cuda") -> None:
        super().__init__()
        self.args = args
        self.eps = args.eps
        self.device = device

        point_opt = args.geoms.points
        pc_feat_opt = args.geoms.point_feats
        bkg_feat_opt = args.geoms.background
        attn_opt = args.models.attn

        _reject_unreachable(args)

        self.use_amp = args.use_amp
        self.amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.pruned_points = False
        self.added_points = False

        self.coord_scale = args.dataset.coord_scale
        self.select_k = int(point_opt.select_k)
        self.pixel_frustum_margin = point_opt.pixel_frustum_margin
        self.topk_dtype = _TOPK_DTYPES[args.topk_dtype]
        self.bkg_sphere_radius = point_opt.get("bkg_sphere_radius", 5.0)
        self.bkg_sphere_center = point_opt.get("bkg_sphere_center", [0.0, 0.0, 0.0])

        self.fx: float | None = None
        self.fy: float | None = None
        self.cx: float | None = None
        self.cy: float | None = None
        self.H: int | None = None
        self.W: int | None = None

        init = initial_points(
            point_opt, coord_scale=self.coord_scale, max_num_pts=args.max_num_pts
        )
        points = init.points
        #: The cloud `load_path` supplied, in the scaled frame. `gt_pcd_loss` chamfers against it.
        self.gt_points = init.gt_points
        self.points = nn.Parameter(points, requires_grad=True)
        num_pts = points.shape[0]

        self.bkg_points: nn.Parameter | None = None
        self.bkg_points_pc_feats: nn.Parameter | None = None
        self.bkg_points_influ_scores: nn.Parameter | None = None
        self.bkg_points_embedv: nn.Parameter | None = None

        if point_opt.influ_init_func == "ones":
            influ = torch.ones(num_pts, 1, device=device) * point_opt.influ_init_val
        elif point_opt.influ_init_func == "randn":
            influ = torch.randn(num_pts, 1, device=device) * point_opt.influ_init_val
        elif point_opt.influ_init_func == "rand":
            influ = torch.rand(num_pts, 1, device=device) + point_opt.influ_init_val
        else:
            raise NotImplementedError(
                f"geoms.points.influ_init_func [{point_opt.influ_init_func}] is not implemented; "
                "expected 'ones', 'randn' or 'rand'"
            )
        self.points_influ_scores = nn.Parameter(influ, requires_grad=True)
        self.points_influ_scores_act = make_activation(point_opt.influ_act)

        self.points_normals = nn.Parameter(torch.randn(num_pts, 3, device=device),
                                           requires_grad=True)

        scaler_init_val = point_opt.get("scaler_init_val", 1.0)
        scaler_learn = point_opt.get("scaler_learn", False)
        scaler_init_func = point_opt.get("scaler_init_func", "ones")
        if scaler_init_func == "randn":
            scaler = torch.randn(num_pts, 1, device=device) * scaler_init_val
        elif scaler_init_func == "rand":
            scaler = (torch.rand(num_pts, 1, device=device) * scaler_init_val
                      + (1.0 - scaler_init_val))
        else:
            scaler = torch.ones(num_pts, 1, device=device) * scaler_init_val
        self.points_scaler = nn.Parameter(scaler, requires_grad=scaler_learn)

        self.points_last_grad = nn.Parameter(torch.zeros(num_pts, 3, device=device),
                                             requires_grad=False)
        self.points_acc_grad = nn.Parameter(torch.zeros(num_pts, 3, device=device),
                                            requires_grad=False)
        self.points_acc_grad_norm = nn.Parameter(torch.zeros(num_pts, device=device),
                                                 requires_grad=False)
        self.points_grad_cnt = nn.Parameter(torch.zeros(num_pts, device=device),
                                            requires_grad=False)

        self.density_k = point_opt.get("density_k", 30)
        self.density_update_interval = point_opt.get("density_update_interval", 1000)
        self.grad_noise_std = point_opt.get("grad_noise_std", 0.0)
        self.points_density = nn.Parameter(torch.ones(num_pts, device=device),
                                           requires_grad=False)
        self.update_point_density()

        self.unet: nn.Module | None = None
        self.fused_feature_mlp: nn.Module | None = None
        self.fused_feature_encoder: nn.Module | None = None
        self.feat_map_dim = attn_opt.output_dim_v
        if args.models.unet.use:
            # The encoder is built for its width alone: with `otype: Identity` it is the identity
            self.fused_feature_encoder = Encoding(attn_opt.output_dim_v,
                                                  args.models.unet.encode_config, True)
            unet_input_dim = self.fused_feature_encoder.n_output_dims
            if args.models.unet.double_channel:
                unet_input_dim *= 2
            self.unet_input_dim = unet_input_dim
            self.unet = unet_from_config(
                args.models.unet, unet_input_dim, out_channels=3,
                use_amp=self.use_amp, amp_dtype=self.amp_dtype,
            )
        elif args.models.fused_feature_mlp.use:
            import tinycudann as tcnn

            self.fused_feature_encoder = Encoding(attn_opt.output_dim_v,
                                                  args.models.fused_feature_mlp.encode_config, True)
            self.fused_feature_mlp = tcnn.Network(
                self.fused_feature_encoder.n_output_dims, 3,
                as_tcnn_config(args.models.fused_feature_mlp.mlp_config),
            )
        elif attn_opt.output_dim_v != 3:
            raise ValueError(
                "with neither models.unet.use nor models.fused_feature_mlp.use, the blended value "
                f"*is* the colour, so models.attn.output_dim_v must be 3, got "
                f"{attn_opt.output_dim_v}"
            )

        self.dumb_constant = 1 / (
            np.exp(bkg_feat_opt.constant) / (np.exp(bkg_feat_opt.constant) + point_opt.select_k)
        )

        self.bkg_feats = nn.Parameter(torch.FloatTensor(bkg_feat_opt.init_color)[None, :],
                                      requires_grad=bkg_feat_opt.learnable)
        bkg_score = torch.tensor(bkg_feat_opt.constant, device=device,
                                 dtype=torch.float32).reshape(1)
        self.bkg_score = nn.Parameter(bkg_score, requires_grad=bkg_feat_opt.learn_score)
        bkg_scaler_init = 1.0 if not args.rnd_background_use_dumb_constant else self.dumb_constant
        self.bkg_scaler = nn.Parameter(torch.ones_like(self.bkg_feats) * bkg_scaler_init,
                                       requires_grad=bkg_feat_opt.learn_scaler)

        self.bkg_token: nn.Parameter | None = None
        if bkg_feat_opt.learn_bkg_token:
            self.bkg_token = nn.Parameter(torch.randn(attn_opt.output_dim_k, device=device),
                                          requires_grad=True)

        num_bkg_feats = 4 if args.rnd_background and args.rnd_background_use_4_feats else 1
        self.append_bkg_points_feats = nn.Parameter(
            torch.randn(num_bkg_feats, pc_feat_opt.dim), requires_grad=True
        )
        self.append_bkg_points_embedv: nn.Parameter | None = None
        if attn_opt.get("append_bkg_points_use_embedv", False):
            init_type = attn_opt.get("append_bkg_points_embedv_init_type", "randn")
            learn = attn_opt.get("append_bkg_points_embedv_learn", True)
            shape = (num_bkg_feats, attn_opt.output_dim_v)
            if init_type == "randn":
                embedv = torch.randn(*shape, device=device)
            elif init_type == "zeros":
                embedv = torch.zeros(*shape, device=device)
            elif init_type == "ones":
                embedv = torch.ones(*shape, device=device)
            else:
                raise NotImplementedError(
                    f"models.attn.append_bkg_points_embedv_init_type [{init_type}] is not "
                    "implemented; expected 'randn', 'zeros' or 'ones'"
                )
            self.append_bkg_points_embedv = nn.Parameter(embedv, requires_grad=learn)

        if attn_opt.use_pc_feats_directly and (pc_feat_opt.dim == 3 or attn_opt.use_sh):
            if attn_opt.use_sh:
                num_bases = (attn_opt.sh_degree + 1) ** 2
                if pc_feat_opt.dim == 3 or pc_feat_opt.dim // num_bases == 3:
                    colors = torch.zeros(num_pts, num_bases, 3)
                    colors[:, 0, :] = rgb_to_sh(torch.rand(num_pts, 3))
                    self.pc_feats = nn.Parameter(colors, requires_grad=True)
                else:
                    self.pc_feats = nn.Parameter(
                        torch.rand(num_pts, num_bases, pc_feat_opt.dim // num_bases),
                        requires_grad=True,
                    )
            else:
                self.pc_feats = nn.Parameter(torch.rand(num_pts, pc_feat_opt.dim),
                                             requires_grad=True)
        else:
            self.pc_feats = nn.Parameter(torch.randn(num_pts, pc_feat_opt.dim), requires_grad=True)

        self.last_act = make_activation(args.models.last_act)
        self.bkg_attn_act = make_activation(args.models.bkg_attn_act)

        attn_args = to_config_node(
            {
                **attn_opt.to_dict(),
                "bkg_sphere_radius": self.bkg_sphere_radius,
                "bkg_sphere_center": self.bkg_sphere_center,
            },
            "models.attn",
        )
        self.proximity_attn = ProximityAttention(
            attn_args,
            point_feats_dim=pc_feat_opt.dim,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            attn_act=args.attn_act,
            attn_act_temp=args.attn_act_temp,
            coord_scale=self.coord_scale,
        )

        self.fix_keys: tuple[str, ...] = tuple(args.training.fix_keys or ())
        self.densification_enabled = not self.fix_keys
        if self.fix_keys:
            print(
                f"[PAPR] training.fix_keys={list(self.fix_keys)} -> point pruning and addition are "
                "DISABLED for this run. A fixed parameter has no optimizer state to resize, and "
                "densifying around one desynchronises the per-point tensors."
            )

        self._check_state_dict_contract()


    def _check_state_dict_contract(self) -> None:
        """Fail now if a key ``models.checkpoint`` requires on save is missing.

        Cheaper than discovering it at the first checkpoint, several hours in.
        """
        present = set(self.state_dict())
        missing = sorted(REQUIRED_ON_SAVE - present)
        if missing:
            raise RuntimeError(
                f"PAPR is missing state_dict keys models.checkpoint requires on save: {missing}"
            )

    def set_camera(self, fx: float, fy: float, cx: float, cy: float, height: int,
                   width: int) -> None:
        """Stamp the dataset's intrinsics on. Required before the first render.

        ``height``/``width`` are the *full image* size even when training on 16x16 patches: the
        frustum test asks whether a point is on screen at all, which a patch cannot answer.
        """
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.H, self.W = int(height), int(width)

    def _require_camera(self) -> None:
        if self.W is None:
            raise RuntimeError(
                "camera intrinsics are not set; call model.set_camera(fx, fy, cx, cy, H, W) with "
                "the dataset's values before rendering"
            )


    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    def point_parameters(self) -> dict[str, nn.Parameter]:
        """The per-point parameters, by name, in the order :mod:`models.densify` must rebuild."""
        return {name: getattr(self, name) for name in PER_POINT_PARAMETERS}

    def set_point_parameters(self, replacements: Mapping[str, nn.Parameter]) -> None:
        """Swap per-point parameters wholesale, checking they stay row-aligned.

        :mod:`models.densify` calls this once per prune or add with *every* per-point tensor. It is
        a single call rather than N assignments because a partial update leaves the cloud in a state
        where ``points[i]`` and ``pc_feats[i]`` describe different points, which renders a plausible
        image instead of crashing.
        """
        unknown = sorted(set(replacements) - set(PER_POINT_PARAMETERS))
        if unknown:
            raise KeyError(f"not per-point parameters: {unknown}")
        missing = sorted(set(PER_POINT_PARAMETERS) - set(replacements))
        if missing:
            raise KeyError(
                f"set_point_parameters needs every per-point tensor at once; missing {missing}"
            )
        counts = {name: int(tensor.shape[0]) for name, tensor in replacements.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"per-point tensors disagree on the point count: {counts}")
        for name, tensor in replacements.items():
            if not isinstance(tensor, nn.Parameter):
                tensor = nn.Parameter(tensor, requires_grad=getattr(self, name).requires_grad)
            setattr(self, name, tensor)

    def get_points(
        self,
        deformed_points: Tensor | None = None,
        drop: bool = False,
        bkg_points_scale: float = 1.0,
        cur_step: int = -1,
    ) -> PointCloud:
        """Assemble the point tensor this step renders.

        Three things can happen to the cloud between ``self.points`` and what the attention scores,
        and all three change the index space ``select_k_ind`` lives in -- which is why the result is
        a :class:`PointCloud` carrying the mask and the dropout indices rather than a bare tensor.

        1. ``deformed_points`` substitutes an externally deformed cloud (the editing path).
        2. ``use_bkg_points`` concatenates a fixed sphere of background points **after** the real
           ones. Refused at construction in this release, but the concatenation is written out
           because it defines the contract: when it fires, index ``i >= len(self.points)`` is a
           background point, and every per-point tensor must be concatenated the same way.
        3. Dropout removes a random fraction of the cloud for this step, so the surviving indices
           address the un-dropped tensor. Every per-point tensor must then be gathered with
           ``kept_indices`` before it is indexed by ``select_k_ind``.

        Args:
            deformed_points: ``(M, 3)`` replacement positions, same shape as ``self.points``.
            drop: Apply per-step point dropout. The caller decides, because ``evaluate`` must not.
            bkg_points_scale: Accepted for signature compatibility; unused, since the
                background-point branch is refused at construction.
            cur_step: Training step, for the dropout-ratio schedule.
        """
        points = self.points
        bkg_points_mask = torch.ones_like(points)[:, 0:1]

        if deformed_points is not None:
            if deformed_points.shape != points.shape:
                raise ValueError(
                    f"deformed_points has shape {tuple(deformed_points.shape)}, expected "
                    f"{tuple(points.shape)}"
                )
            points = deformed_points

        if self.args.geoms.points.use_bkg_points:
            points = torch.cat([points, self.bkg_points], dim=0)
            bkg_points_mask = torch.cat(
                [bkg_points_mask, torch.zeros_like(self.bkg_points)[:, 0:1]], dim=0
            )

        kept_indices: Tensor | None = None
        if drop:
            max_ratio = self.get_drop_points_max_ratio(cur_step)
            if max_ratio > 0:
                ratio = random.uniform(0, max_ratio)
                num_points = points.shape[0]
                num_keep = int(num_points * (1 - ratio))
                if num_keep < num_points:
                    kept_indices = torch.randperm(num_points, device=points.device)[:num_keep]
                    points = points[kept_indices]
                    bkg_points_mask = bkg_points_mask[kept_indices]

        return PointCloud(points=points, bkg_points_mask=bkg_points_mask,
                          kept_indices=kept_indices)

    def get_drop_points_max_ratio(self, cur_step: int = -1) -> float:
        """Current ceiling on the per-step dropout fraction.

        Dropout is a regulariser against the cloud over-fitting a fixed shortlist: with 30% of the
        points gone each step, a surface cannot come to depend on one particular point being in
        every ray's top-k. It anneals to zero so the final model is trained on the cloud it will be
        evaluated with.
        """
        point_opt = self.args.geoms.points
        max_ratio = point_opt.get("drop_points_max_ratio", -1.0)
        if max_ratio <= 0:
            return 0.0

        drop_start_step = point_opt.get("drop_points_start_step", 0)
        if 0 <= cur_step < drop_start_step:
            return 0.0
        if cur_step < 0:
            return max_ratio

        schedule_type = point_opt.get("drop_points_max_ratio_schedule_type", "none")
        if schedule_type == "none":
            return max_ratio

        start_step = point_opt.get("drop_points_max_ratio_start_step", 0)
        end_step = point_opt.get("drop_points_max_ratio_end_step", -1)
        if end_step < 0:
            end_step = self.args.training.steps
        if cur_step < start_step:
            return max_ratio
        if cur_step >= end_step:
            return 0.0

        span = max(float(end_step - start_step), 1.0)
        t = (cur_step - start_step) / span
        if schedule_type == "cosine":
            return float(max_ratio * 0.5 * (1.0 + np.cos(t * np.pi)))
        if schedule_type == "linear":
            return float(max_ratio * (1.0 - t))
        raise NotImplementedError(
            f"geoms.points.drop_points_max_ratio_schedule_type [{schedule_type}] is not "
            "implemented; expected 'none', 'cosine' or 'linear'"
        )

    def get_pc_feats(self, bkg_color: Tensor | None = None,
                     kept_indices: Tensor | None = None) -> Tensor:
        """Per-point features aligned with :meth:`get_points`.

        ``kept_indices`` comes from the :class:`PointCloud` of the same call. Passing it explicitly
        rather than reading a ``self.kept_indices`` written by ``get_points`` is the same fix as
        with dropout on, a features tensor gathered from a *different* step's mask is
        silently misaligned with the point positions.
        """
        pc_feats = self.pc_feats
        if self.args.geoms.points.use_bkg_points:
            bkg_pc_feats = self.bkg_points_pc_feats.expand(self.bkg_points.shape[0], -1)
            if pc_feats.dim() != bkg_pc_feats.dim():
                bkg_pc_feats = bkg_pc_feats.reshape(self.pc_feats.shape)
            pc_feats = torch.cat([pc_feats, bkg_pc_feats], dim=0)
        if kept_indices is not None:
            pc_feats = pc_feats[kept_indices]
        return pc_feats

    def get_influ_scores(self, kept_indices: Tensor | None = None) -> Tensor:
        """Per-point influence scores aligned with :meth:`get_points`."""
        scores = self.points_influ_scores
        if self.args.geoms.points.use_bkg_points:
            bkg = self.bkg_points_influ_scores.expand(self.bkg_points.shape[0], -1)
            scores = torch.cat([scores, bkg], dim=-2)
        if kept_indices is not None:
            scores = scores[kept_indices]
        return scores

    def get_scaler(self, kept_indices: Tensor | None = None) -> Tensor:
        """Per-point geometry scale aligned with :meth:`get_points`."""
        scaler = self.points_scaler
        if self.args.geoms.points.use_bkg_points:
            bkg = torch.ones(self.bkg_points.shape[0], 1, device=self.points_scaler.device)
            scaler = torch.cat([scaler, bkg], dim=-2)
        if kept_indices is not None:
            scaler = scaler[kept_indices]
        return scaler

    def get_bkg_sphere_intersection(self, rays_o: Tensor, rays_d: Tensor,
                                    sphere_center: Any = None,
                                    sphere_radius: float | None = None) -> Tensor:
        """Where each ray leaves the background sphere, in the renderer's scaled frame.

        Delegates to the attention module so there is exactly one implementation and one sphere.
        The ``coord_scale`` multiplication happens inside ``ProximityAttention``, which is why the
        overrides here are given in scene units like the config values they replace.
        """
        if sphere_radius is not None:
            sphere_radius = sphere_radius * self.coord_scale
        if sphere_center is not None:
            sphere_center = [c * self.coord_scale for c in sphere_center]
        return self.proximity_attn.get_bkg_sphere_intersection(
            rays_o, rays_d, sphere_center=sphere_center, sphere_radius=sphere_radius
        )

    @torch.no_grad()
    def update_point_density(self) -> None:
        """Refresh ``points_density`` = 1 / (distance to the k-th nearest neighbour).

        Local density is what the point-addition heuristic reads: new points go where the cloud is
        sparse relative to the gradient signal there. Computed on the CPU with a KD-tree because
        the query is over the whole cloud against itself, runs a few dozen times per run, and a
        GPU KNN over 60k points costs more to set up than it saves.
        """
        from scipy.spatial import cKDTree

        n_points = self.points.shape[0]
        if n_points < self.density_k:
            self.points_density.data.fill_(1.0)
            return

        device = self.points.device
        points_np = (self.points.detach() / self.coord_scale).cpu().numpy()
        tree = cKDTree(points_np)
        distances, _ = tree.query(points_np, k=self.density_k, workers=-1)
        kth = np.maximum(distances[:, -1], 1e-10)
        self.points_density.data = torch.from_numpy(1.0 / kth).float().to(device)


    def get_attn_weights(self, scores: Tensor) -> Tensor:
        """Normalise the per-slot scores into attention weights.

        Reduced to the one reachable branch:
        ``attn_act: softmax`` in every shipped config, and ``bkg_exp_scaler: false``, so the
        learned background-exponential variant -- which divides by ``sum + eps`` where plain
        softmax does not, and is therefore *not* a special case of it -- never applies.
        ``ProximityAttention`` refuses any other ``attn_act`` at construction.
        """
        return attention_weights(scores, self.args.attn_act_temp)

    def get_append_bkg_points_feats(self, bkg_color: Tensor | None = None) -> Tensor:
        """The ``(1, D)`` feature the appended background slot carries.

        With ``rnd_background`` on and four feature vectors, the vectors are combined with the
        step's random background colour as weights, so the slot's value tracks the colour the
        target image was composited against. With one vector (all nine configs) it is returned
        unchanged and ``bkg_color`` is ignored.

        ``evaluate`` passes ``bkg_color=None``, which makes the four-vector case a plain unweighted
        sum, while ``forward`` weights it by the step's background colour. That asymmetry is
        preserved rather than unified: it is only reachable under ``rnd_background``,
        which no shipped config enables, and unifying it would change what such a run renders.
        """
        feats = self.append_bkg_points_feats
        if feats.shape[0] == 4:
            weights = torch.ones(4, device=feats.device)
            if bkg_color is not None:
                weights[1:] = bkg_color.squeeze()
            return torch.sum(feats * weights.unsqueeze(-1), dim=0, keepdim=True)
        return feats

    def select_topk(self, points: Tensor, c2w: Tensor, pix_coords: Tensor) -> TopkSelection:
        """Shortlist ``select_k`` candidate points per ray by pixel-space proximity.

        Returns the selection rather than writing ``self.select_k_ind``. The
        projection is done here (not in :mod:`models.topk`) because the intrinsics live on the
        model and the same projection feeds the frustum test's depth.
        """
        self._require_camera()
        with torch.autocast(device_type="cuda", enabled=False):
            points_2d, points_cam = fused_projection(points.float(), c2w.float(), self.fx, self.fy,
                                                     self.cx, self.cy, self.W)
        z = points_cam[..., 2]
        if z.ndim == 2:
            z = z[:, None, None, :]
        return select_topk_points(
            points_2d,
            pix_coords,
            z,
            select_k=self.select_k,
            image_height=self.H,
            image_width=self.W,
            select_k_type=self.args.geoms.points.select_k_type,
            select_k_z_pow=self.args.geoms.points.select_k_z_pow,
            pixel_frustum_margin=self.pixel_frustum_margin,
            topk_dtype=self.topk_dtype,
            eps=self.eps,
        )

    def _apply_influence(self, scores: Tensor, influ: Tensor | None,
                         select_k_ind: Tensor) -> Tensor:
        """Fold the selected points' influence scores into the raw attention scores."""
        if influ is None:
            return scores
        return apply_influence_scores(scores, influ[select_k_ind],
                                      self.args.influ_scores_fuse_type)

    def _substitute_bkg_embedv(self, values: Tensor, bkg_color: Tensor | None) -> Tensor:
        """Replace the background slot's value with its own learned embedding.

        The value network produces a vector for the background slot like any other, computed from
        the ray/sphere-intersection geometry. When ``append_bkg_points_use_embedv`` is set (all
        nine configs) that vector is discarded in favour of a free parameter, which lets the
        background be a flat learned colour independent of where on the sphere the ray exits.

        Written with ``torch.where`` rather than an in-place slice assignment because ``values``
        is an autograd intermediate.
        """
        if self.append_bkg_points_embedv is None:
            return values
        embedv = self.append_bkg_points_embedv
        if embedv.ndim == 2:
            if embedv.shape[0] == 4:
                weights = torch.ones(4, device=embedv.device)
                if bkg_color is not None:
                    weights[1:] = bkg_color.squeeze()
                embedv = torch.sum(embedv * weights.unsqueeze(-1), dim=0)
            else:
                embedv = embedv.squeeze(0)
        n, h, w = values.shape[:3]
        slot_mask = torch.zeros(values.shape[:-1], dtype=torch.bool, device=values.device)
        slot_mask[..., -1] = True
        bkg_slot = embedv.expand(n, h, w, -1).unsqueeze(-2)
        return torch.where(slot_mask.unsqueeze(-1), bkg_slot, values)


    def _render_slots(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        c2w: Tensor,
        pix_coords: Tensor,
        bkg_color: Tensor | None,
        cur_step: int,
        deformed_points: Tensor | None,
        pc_feats: Tensor | None,
        drop: bool,
    ):
        """Everything both render paths share: cloud, shortlist, scores, weights, values.

        ``forward`` and ``evaluate`` differ only in whether dropout is applied and in the shape
        they blend to, so the ~40 lines between the cloud and the attention weights live here
        once. ``forward`` weights the background feature by the step's background colour and
        ``evaluate`` does not: ``evaluate`` calls this with ``bkg_color=None``, which is exactly
        the unweighted sum. Sharing the body preserves that difference instead of hiding it; see
        :meth:`get_append_bkg_points_feats`.
        """
        cloud = self.get_points(deformed_points, drop=drop, cur_step=cur_step)
        if pc_feats is None:
            pc_feats = self.get_pc_feats(bkg_color, kept_indices=cloud.kept_indices)
        influ = self.get_influ_scores(kept_indices=cloud.kept_indices)
        scaler = self.get_scaler(kept_indices=cloud.kept_indices)

        selection = self.select_topk(cloud.points, c2w, pix_coords)
        select_k_ind = selection.indices

        out = self.proximity_attn(
            rays_o, rays_d, cloud.points, pc_feats, select_k_ind,
            append_bkg_points_feats=self.get_append_bkg_points_feats(bkg_color),
            points_scaler=scaler,
            step=cur_step,
        )

        n, h, w, _ = rays_d.shape
        values = out.values.reshape(n, h, w, -1, out.values.shape[-1])
        scores = out.scores.reshape(n, h, w, -1, 1)
        scores = self._apply_influence(scores, influ, select_k_ind)

        attn = self.get_attn_weights(scores)
        values = self._substitute_bkg_embedv(values, bkg_color)
        return cloud, select_k_ind, selection.min_d2r, attn, values, out

    def forward(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        c2w: Tensor,
        pix_coords: Tensor,
        bkg_color: Tensor | None = None,
        cur_step: int = -1,
        *,
        deformed_points: Tensor | None = None,
        pc_feats: Tensor | None = None,
    ) -> RenderOutput:
        """Render a batch of ray patches, with gradients.

        Args:
            rays_o: ``(N, 3)`` ray origins, one per view.
            rays_d: ``(N, H, W, 3)`` unit ray directions.
            c2w: ``(N, 4, 4)`` camera-to-world matrices, coord-scale premultiplied.
            pix_coords: ``(H, W, 2)`` or ``(N, H, W, 2)`` pixel centres of this patch.
            bkg_color: ``(1, 3)`` background colour this step was composited against. Only read by
                the four-feature background path and the alpha blend, neither reachable from the
                shipped configs; pass it anyway so ``rnd_background`` runs stay correct.
            cur_step: Training step, for the dropout schedule and the SH band schedule.
            deformed_points: ``(M, 3)`` replacement point positions (the editing path).
            pc_feats: ``(M, D)`` replacement point features (the UV-texture path).

        Returns:
            A :class:`RenderOutput`. Apply ``model.last_act`` to ``rgb`` before the loss; it is
            not applied here, because ``evaluate`` does not apply it either and the two must
            composite identically.
        """
        cloud, select_k_ind, min_d2r, attn, values, out = self._render_slots(
            rays_o, rays_d, c2w, pix_coords, bkg_color, cur_step, deformed_points, pc_feats,
            drop=self.args.geoms.points.drop_points_max_ratio > 0,
        )
        n, h, w, _ = rays_d.shape

        topk_attn = attn[..., :-1, :]
        bkg_attn = attn[..., -1, :]

        fused_features = torch.sum(values * attn, dim=3)
        rgb, fused_features = self.decode(fused_features, n, h, w)

        if self.args.models.attn.get("append_bkg_points_alpha_blend", False):
            effective_bkg_color = bkg_color.expand(n, h, w, -1)
            rgb = rgb * (1 - bkg_attn) + effective_bkg_color * bkg_attn

        return RenderOutput(
            rgb=rgb,
            fused_features=fused_features,
            attn=attn,
            topk_attn=topk_attn,
            bkg_attn=bkg_attn,
            select_k_ind=select_k_ind,
            cloud=cloud,
            min_d2r=min_d2r,
            sphere_intersection=out.sphere_intersection,
            pd=out.pd,
            d2r=out.d2r,
            values=values,
        )

    def decode(self, fused_features: Tensor, n: int, h: int, w: int) -> tuple[Tensor, Tensor]:
        """Blended features to RGB. Returns ``(rgb, fused_features)``.

        ``fused_features`` is returned as well because the ``fused_feature_mlp`` branch *replaces*
        it with its encoded form before decoding, and that reassigned form is what the caller's
        diagnostics see.

        Public, and deliberately so. :meth:`forward` decodes its own patch, but the tiled render
        paths in ``test.py``, ``train.py`` and ``edit/render.py`` accumulate an
        ``(N, H, W, 1, C)`` frame buffer from :meth:`evaluate` and decode it once at the end --
        a per-tile decode would put a U-Net receptive-field boundary at every tile seam. Those
        four callers must run the identical branch, which is why this is one method rather than
        four copies: duplicating it is how the training, test and edit renders would drift apart.
        """
        if self.args.models.unet.use:
            if self.args.models.unet.double_channel:
                fused_features = torch.cat([fused_features, fused_features], dim=-1)
            rgb = self.unet(fused_features.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            return rgb, fused_features
        if self.args.models.fused_feature_mlp.use:
            encoded = self.fused_feature_encoder(fused_features.reshape(n * h * w, -1))
            rgb = self.fused_feature_mlp(encoded).reshape(n, h, w, 3)
            return rgb, encoded.reshape(n, h, w, -1)
        return fused_features, fused_features

    @torch.no_grad()
    def evaluate(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        c2w: Tensor,
        pix_coords: Tensor,
        step: int = -1,
        *,
        bkg_color: Tensor | None = None,
        deformed_points: Tensor | None = None,
        pc_feats: Tensor | None = None,
    ) -> EvalOutput:
        """Render one tile without gradients, stopping before the decoder.

        Full-resolution frames do not fit in memory as one attention batch, so the caller renders
        tiles and assembles an ``(N, H, W, 1, C)`` feature frame, then runs the U-Net **once** over
        the whole frame. Decoding per tile would put a receptive-field discontinuity at every seam.
        That is why this returns features rather than colour, and why ``fused_features`` keeps its
        singleton slot axis: it is a slice assignment target.

        Point dropout is never applied here: evaluation renders the cloud the model actually has.

        Everything the caller needs to index its frame buffer -- ``select_k_ind``, the cloud those
        indices address, ``pd``/``d2r``/``sphere_intersection`` -- is returned rather than left on
        the module, because the caller's next statement writes it into a full-frame
        buffer at this tile's offset.
        """
        cloud, select_k_ind, min_d2r, attn, values, out = self._render_slots(
            rays_o, rays_d, c2w, pix_coords, bkg_color, step, deformed_points, pc_feats,
            drop=False,
        )
        fused_features = torch.sum(values * attn, dim=3, keepdim=True)
        return EvalOutput(
            fused_features=fused_features,
            attn=attn,
            select_k_ind=select_k_ind,
            cloud=cloud,
            min_d2r=min_d2r,
            sphere_intersection=out.sphere_intersection,
            pd=out.pd,
            d2r=out.d2r,
            values=values,
        )


    def get_scheduled_weight(self, base_weight: float, step: int, start: float, stop: float,
                             schedule_type: str = "constant") -> float:
        """Weight of a regularizer at ``step``, zero outside ``[start, stop]``.

        Regularizers here are scaffolding, not objectives: the point-on-ray and point-to-surface
        terms pull the cloud onto the surface early and must be out of the way before the model
        fits high-frequency texture, or they flatten it. The window and the ramp are how that is
        expressed.

        ``schedule_type`` is one of ``constant``, ``cosine``/``cosine_decay`` (1 -> 0 over the
        window) or ``cosine_warmup`` (0 -> 1). An unrecognised name falls through to ``constant``:
        a typo in a config silently gets a constant weight rather than an error.

        Delegates to :func:`models.regularizers.scheduled_weight` so the release has exactly one
        implementation. It matters: the expression evaluates the cosine in **float32**
        (``torch.cos(torch.tensor(t) * torch.pi)``), and a ``numpy`` version of the same formula
        computes it in float64 and differs in the last few bits of every non-trivial weight.
        """
        return scheduled_weight(base_weight, step, start, stop, schedule_type)

    def compute_scheduled_loss(self, loss_name: str, cur_step: int, loss_fn,
                               extra_condition: bool = True) -> tuple[Tensor, float]:
        """Evaluate one regularizer, but only if its scheduled weight is non-zero.

        ``loss_fn`` is a zero-argument callable rather than a tensor so that a disabled term costs
        nothing: several of these run a chamfer distance or a KNN over the whole cloud.

        The zero returned when the term is off is ``torch.zeros(0).sum()`` -- a real
        graph-connected scalar zero, so the caller can sum it into the loss unconditionally.

        Args:
            loss_name: Config prefix, e.g. ``point_on_ray_loss``. Reads
                ``training.{name}_weight``, ``_start``, ``_stop`` and ``_schedule``, falling back
                to ``training.regularizer_schedule``.
            cur_step: Training step.
            loss_fn: Called only when the weight is positive and ``extra_condition`` holds.
            extra_condition: An additional gate. ``gt_sp_loss`` passes ``self.pruned_points``
                here, which is the gate that makes a fine-tune optimise a different objective from
                a fresh run for its first ``prune_start`` steps -- see ``__init__``.
        """
        return scheduled_loss(
            loss_name, cur_step, loss_fn,
            training=self.args.training,
            device=self.points.device,
            extra_condition=extra_condition,
        )


    def save(self, step: int, save_dir: str | Path, *,
             optimizers: Mapping[str, Any] | None = None,
             schedulers: Mapping[str, Any] | None = None) -> dict[str, Path]:
        """Write ``model.pth`` plus the optimizer/scheduler/scaler sidecars.

        Delegates to :func:`models.checkpoint.save_checkpoint`, which owns the container format and
        checks the state-dict contract. ``optimizers`` and ``schedulers`` are already-serialised
        state dicts from :mod:`models.optim`; the model no longer builds its own optimizers, so it
        cannot produce them itself.
        """
        save_dir = Path(save_dir)
        extra: dict[str, Any] = {"scaler": self.scaler.state_dict()}
        if optimizers is not None:
            extra["optimizers"] = dict(optimizers)
        if schedulers is not None:
            extra["schedulers"] = dict(schedulers)
        return save_checkpoint(save_dir / "model.pth", self, step, extra=extra)

    def load(self, path: str | Path, *, strict: bool = True, require_step: bool = True):
        """Restore weights from ``path``. Returns the :class:`~models.checkpoint.LoadReport`.

        Per-point parameters are resized to the checkpoint's point count first -- a trained model
        rarely has the same N it started with. Restoring optimizers, schedulers and
        ``self.scaler`` is the caller's job, because it is the caller that built them.

        The caller must also set ``model.pruned_points = True`` when this is a ``--load_path``
        fine-tune rather than a ``--resume``; see the note in ``__init__``.
        """
        report = load_into_model(self, path, strict=strict, require_step=require_step)
        self._realign_omitted_point_parameters()
        return report

    def _realign_omitted_point_parameters(self) -> None:
        """Resize any per-point tensor the checkpoint did not carry to the loaded point count.

        ``points_normals`` post-dates most archived checkpoints, so loading one leaves it at the
        *initial* N while ``points`` has grown to the trained N. Nothing reads it in the shipped
        configs -- the normal-consistency loss that would is weighted 0.0 everywhere -- so the
        image is unaffected either way, but a per-point tensor whose rows do not correspond to
        ``points`` rows is a trap for :mod:`models.densify`, which reindexes all of them together.

        The replacement is drawn on a *private* CPU generator, never the global CUDA stream: drawing
        ``torch.randn(N, 3, device="cuda")`` here would advance the CUDA generator and change every
        subsequent ``torch.randperm`` -- which is exactly how point dropout is sampled. Five of the
        nine shipped configs both load a checkpoint and enable dropout, so a global draw would give
        a seeded fine-tune a different sequence of dropped clouds from step 0. The value is
        arbitrary either way (no checkpoint ever recorded one); the stream position is not.
        """
        count = self.points.shape[0]
        for name in PER_POINT_PARAMETERS:
            param = getattr(self, name)
            if param.shape[0] == count:
                continue
            shape = (count,) + tuple(param.shape[1:])
            if name == "points_normals":
                generator = torch.Generator(device="cpu").manual_seed(_NORMALS_SEED)
                fresh = torch.randn(*shape, generator=generator, dtype=param.dtype)
                fresh = fresh.to(device=param.device)
            else:
                fresh = torch.zeros(*shape, device=param.device, dtype=param.dtype)
            if name in ("points_density", "points_scaler"):
                fresh = torch.ones(*shape, device=param.device, dtype=param.dtype)
            setattr(self, name, nn.Parameter(fresh, requires_grad=param.requires_grad))
            print(f"[PAPR] {name} was absent from the checkpoint; re-initialised at N={count}")


#: Fixed seed for the private generator used to refill ``points_normals``. Kept off the
#: global stream so a checkpoint load consumes no randomness. See
#: ``PAPR._realign_omitted_point_parameters``.
_NORMALS_SEED = 0


def _reject_unreachable(args: Any) -> None:
    """Fail at construction for any config selecting a path this release does not implement.

    Each of these changes what the renderer computes. Refusing at construction rather than
    ignoring the knob is the difference between a config that does not run and a config that runs
    and produces something the released checkpoints were never trained to produce.
    """
    point_opt = args.geoms.points
    checks: list[tuple[bool, str, str]] = [
        (
            args.geoms.no_additional_bkg,
            "geoms.no_additional_bkg",
            "renders with no background slot at all; every shipped config composites a background",
        ),
        (
            point_opt.use_bkg_points,
            "geoms.points.use_bkg_points",
            "concatenates a second background point cloud into the index space. get_points() "
            "still writes the concatenation out as the contract, but the renderer, the attention "
            "module and the surface fusion have no background-mask parameter",
        ),
        (
            args.geoms.get("append_sphere_bkg_points", False),
            "geoms.append_sphere_bkg_points",
            "appends sphere points to the optimised cloud at init",
        ),
        (
            point_opt.select_k_rnd,
            "geoms.points.select_k_rnd",
            "resamples select_k every step, which changes the attention's slot count per step",
        ),
        (
            point_opt.add_noise,
            "geoms.points.add_noise",
            "jitters the point positions each step",
        ),
        (
            args.geoms.rays.perturb,
            "geoms.rays.perturb",
            "jitters the ray directions and resamples the target with grid_sample",
        ),
        (
            args.geoms.alpha.use,
            "geoms.alpha.use",
            "adds a per-point alpha with its own supervision",
        ),
        (
            args.exposure_control.use,
            "exposure_control.use",
            "adds the IMLE mapping MLP and per-image shading codes",
        ),
        (
            args.texture.use_v or args.texture.use_combine_network,
            "texture.use_v / texture.use_combine_network",
            "adds the texture value network",
        ),
        (
            args.models.attn_type != "proximity",
            "models.attn_type",
            f"is {args.models.attn_type!r}; only 'proximity' is implemented "
            "(the patch_gauss and self-attention renderers need kernels this release omits)",
        ),
        (
            args.models.get("refiner", None) is not None and args.models.refiner.use,
            "models.refiner.use",
            "adds the local residual refiner",
        ),
        (
            args.models.get("dual_value", None) is not None and args.models.dual_value.use,
            "models.dual_value.use",
            "adds the dual value head",
        ),
        (
            args.models.attn.get("factorized_score", None) is not None
            and args.models.attn.factorized_score.get("use", False),
            "models.attn.factorized_score.use",
            "replaces the key network with per-point tables and needs the fused CUDA kernels",
        ),
        (
            args.models.normalize_topk_attn,
            "models.normalize_topk_attn",
            "renormalises the foreground weights by (1 - bkg_attn); only reachable together with "
            "the separate-background-token compositing branch, which this release drops",
        ),
        (
            args.rescale_bkg_points_for_attn,
            "rescale_bkg_points_for_attn",
            "rescales background points before scoring",
        ),
        (
            "mlp" in point_opt.select_k_type,
            "geoms.points.select_k_type",
            "selects the learned top-k MLP selector",
        ),
    ]
    cubemap = args.get("cubemap_texture", None)
    if cubemap is not None:
        checks.append((
            cubemap.get("use_rgb", False) or cubemap.get("use_feature_map", False)
            or cubemap.get("use_depth_map", False),
            "cubemap_texture.use_*",
            "renders the background from a learned cubemap",
        ))

    offenders = [f"  - {key}: {why}" for enabled, key, why in checks if enabled]
    if offenders:
        raise NotImplementedError(
            "this release implements the single render path the nine shipped configs use; the "
            "following config keys select something else:\n" + "\n".join(offenders)
        )

from __future__ import annotations

import math
from typing import Any, NamedTuple

import tinycudann as tcnn
import torch
from torch import Tensor, nn

from models.activations import make_activation
from models.features import (
    Encoding,
    as_tcnn_config,
    get_tcnn_init_weights,
    key_features,
    kqv_dims,
    query_features,
    ray_point_geometry,
    value_features,
)
from models.sh import active_sh_degree, spherical_harmonics
from utils.geometry import ray_sphere_intersection

__all__ = [
    "AttentionOutput",
    "DEFAULT_BKG_SPHERE_CENTER",
    "DEFAULT_BKG_SPHERE_RADIUS",
    "LayerNorm",
    "ProximityAttention",
    "apply_influence_scores",
    "attention_weights",
]

#: Sphere the background slot's key point is placed on when the config does not stamp one on.
DEFAULT_BKG_SPHERE_RADIUS = 5.0
DEFAULT_BKG_SPHERE_CENTER = (0.0, 0.0, 0.0)


class LayerNorm(nn.Module):
    """Layer normalisation over the last dimension.

    Two deliberate differences from :class:`torch.nn.LayerNorm`, both load-bearing:

    * the affine scale is named ``weights``, not ``weight``. Every archived checkpoint spells the
      key ``proximity_attn.model_k.1.weights``; renaming it strands all 477 of them.
    * the denominator is ``sqrt(var + eps)`` using the *unbiased* variance
      (:meth:`torch.Tensor.var` with its default ``correction=1``), where torch's uses the biased
      one. On the 39- and 128-wide vectors this module normalises the two differ by a factor
      ``sqrt(n / (n - 1))`` -- about 1.3% at n=39 -- which the affine scale has absorbed over
      training. It is not a bug to fix; it is what the weights were fitted against.

    Computing the variance rather than the standard deviation is also deliberate: ``std()`` has an
    infinite gradient at zero spread, which a constant input hits exactly, and the query features
    of every released config (``q_type: 5``) are constant.
    """

    def __init__(self, feat_dim: int, elementwise_affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weights = nn.Parameter(torch.ones(feat_dim))
            self.bias = nn.Parameter(torch.zeros(feat_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            return self.weights * (x - mean) / std + self.bias
        return (x - mean) / std

    def extra_repr(self) -> str:
        return f"elementwise_affine={self.elementwise_affine}, eps={self.eps}"


def _build_network(
    input_dims: int,
    output_dims: int,
    network_config: Any,
    init_output_dims: int | None = None,
) -> nn.Module:
    """A tinycudann network with a Xavier initialisation written over its parameters.

    ``init_output_dims`` defaults to ``output_dims``; the key network passes ``output_dims - 1``,
    which leaves its last output channel at exactly zero. See
    :func:`~models.features.get_tcnn_init_weights` for why that is preserved rather than corrected.
    """
    network = tcnn.Network(input_dims, output_dims, as_tcnn_config(network_config))
    init_dims = output_dims if init_output_dims is None else init_output_dims
    network.params.data[...] = get_tcnn_init_weights(input_dims, init_dims, network_config)
    return network


def _projection(dim: int) -> nn.Linear:
    """A square Xavier-initialised projection. Its bias keeps ``nn.Linear``'s default draw."""
    layer = nn.Linear(dim, dim)
    nn.init.xavier_uniform_(layer.weight)
    return layer


class AttentionOutput(NamedTuple):
    """What :meth:`ProximityAttention.forward` produces.

    ``select_k_ind`` is **returned**, never stashed on the module. Writing it to
    ``self.select_k_ind`` and reading it back a statement or two later risks pairing one tile's
    attention with another tile's indices the moment anything tiles or reorders the render. The same
    applies to ``sphere_intersection``, ``pd`` and ``d2r``, which the surface-point fusion and the
    eval diagnostics read off the returned bundle rather than off the module.
    """

    #: ``(N*H*W, K+1)`` activated scores, background slot last. Pre-softmax.
    scores: Tensor
    #: ``(N*H*W*(K+1), D_v)``, or ``(N, H, W, K+1, D)`` on the spherical-harmonic value path.
    values: Tensor
    #: ``(N, H, W, K)`` point indices these scores and values belong to.
    select_k_ind: Tensor
    #: ``(N, H, W, 1, 3)`` where each ray leaves the background sphere.
    sphere_intersection: Tensor
    #: ``(N, H, W, K+1, 1)`` unsigned along-ray depth of every slot.
    pd: Tensor
    #: ``(N, H, W, K+1, 1)`` perpendicular distance from every slot to its ray.
    d2r: Tensor


class ProximityAttention(nn.Module):
    """Score and blend the ``select_k`` points nearest each ray.

    Args:
        args: The ``models.attn`` config block. The model stamps ``bkg_sphere_radius`` and
            ``bkg_sphere_center`` onto it so the background slot and the surface-point fusion agree
            on one sphere.
        point_feats_dim: Width of a point's learned feature. Recorded for callers; the value
            network's own input width comes from ``encode_additional_dim_v``.
        use_amp: Recorded for callers. The module's autocast handling is the unconditional fp32
            island in :meth:`forward`, not a function of this flag.
        amp_dtype: Likewise recorded, not applied here.
        attn_act: Attention nonlinearity. Checked here so an unsupported one fails at construction
            rather than at the first :func:`attention_weights` call.
        attn_act_temp: Softmax temperature, applied by :func:`attention_weights`.
        coord_scale: The scene's coordinate scale. The sphere radius and centre are given in scene
            units and multiplied by it here, once.
    """

    def __init__(
        self,
        args: Any,
        point_feats_dim: int = 64,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        attn_act: str = "softmax",
        attn_act_temp: float = 1.0,
        coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        _reject_unreachable(args)

        self.args = args
        self.point_feats_dim = int(point_feats_dim)
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.attn_act = attn_act
        self.attn_act_temp = attn_act_temp
        self.coord_scale = coord_scale
        if attn_act != "softmax":
            raise NotImplementedError(
                f"attn_act={attn_act!r} is not implemented in this release; every shipped config "
                "uses 'softmax'"
            )

        radius = args.get("bkg_sphere_radius", None)
        self.bkg_sphere_radius = (
            DEFAULT_BKG_SPHERE_RADIUS if radius is None else radius
        ) * coord_scale
        center = args.get("bkg_sphere_center", None)
        if center is None:
            center = DEFAULT_BKG_SPHERE_CENTER
        self.bkg_sphere_center = [c * coord_scale for c in center]

        dims = self.get_kqv_dim()
        self.input_dim_k, self.input_dim_q, self.input_dim_v = dims.k, dims.q, dims.v

        self.model_k = nn.Sequential(*self._kq_stack(
            self.input_dim_k, args.encode_k_config, args.network_k_config, args.output_dim_k,
            norm_in=args.norm_in_k, norm_out=args.norm_out_k, project=args.project_k,
            init_output_dims=args.output_dim_k - 1,
        ))
        self.model_q = nn.Sequential(*self._kq_stack(
            self.input_dim_q, args.encode_q_config, args.network_q_config, args.output_dim_q,
            norm_in=args.norm_in_q, norm_out=args.norm_out_q, project=args.project_q,
            init_output_dims=args.output_dim_q,
        ))

        encoder_v = Encoding(self.input_dim_v, args.encode_v_config, args.use_tcnn_encoder)
        value_modules: list[nn.Module] = [encoder_v]
        if args.network_v:
            value_in = encoder_v.n_output_dims + args.encode_additional_dim_v
            value_modules.append(_build_network(value_in, args.output_dim_v, args.network_v_config))
        self.model_v = nn.ModuleList(value_modules)

        self.score_act = make_activation(args.score_act)
        self.score_scale = nn.Parameter(
            torch.tensor(args.score_scale, dtype=torch.float32),
            requires_grad=args.score_scale_learnable,
        )


    def _kq_stack(
        self,
        input_dim: int,
        encode_config: Any,
        network_config: Any,
        output_dim: int,
        *,
        norm_in: str,
        norm_out: str,
        project: bool,
        init_output_dims: int,
    ) -> list[nn.Module]:
        """The five-module key/query stack whose indices the archived keys pin down.

        ``0`` encoding, ``1`` input LayerNorm, ``2`` tinycudann network, ``3`` output LayerNorm,
        ``4`` projection. Each of the three optional pieces is individually switchable in the config
        and enabled in every released config; switching one off shifts every index after it, so an
        unset one is refused rather than silently renumbering the module.
        """
        encoder = Encoding(input_dim, encode_config, self.args.use_tcnn_encoder)
        if norm_in != "layernorm":
            raise NotImplementedError(
                f"norm_in={norm_in!r} is not implemented; every shipped config uses 'layernorm', "
                "and omitting the module renumbers every archived state_dict key after it"
            )
        modules: list[nn.Module] = [encoder, LayerNorm(encoder.n_output_dims)]

        if network_config.otype == "MLP":
            raise NotImplementedError(
                "network otype 'MLP' is not implemented in this release; every shipped config uses "
                "a tinycudann backend (FullyFusedMLP / CutlassMLP)"
            )
        modules.append(_build_network(
            encoder.n_output_dims, output_dim, network_config, init_output_dims=init_output_dims
        ))

        if norm_out != "layernorm":
            raise NotImplementedError(
                f"norm_out={norm_out!r} is not implemented; every shipped config uses 'layernorm'"
            )
        modules.append(LayerNorm(output_dim))
        if not project:
            raise NotImplementedError(
                "project_k / project_q are true in every shipped config; disabling one drops the "
                "final Linear and renumbers the archived state_dict keys"
            )
        modules.append(_projection(output_dim))
        return modules

    def get_kqv_dim(self):
        """Widths of the raw key, query and value features, before positional encoding.

        Returns a :class:`~models.features.KQVDims` rather than assigning ``self.input_dim_*`` as a
        side effect, so a config whose ``*_type`` and network widths disagree fails here instead of
        at the first forward.
        """
        return kqv_dims(self.args.k_type, self.args.q_type, self.args.v_type)


    def get_bkg_sphere_intersection(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        sphere_center: Any = None,
        sphere_radius: float | None = None,
    ) -> Tensor:
        """Where each ray leaves the background sphere, shaped like ``rays_d``.

        This point becomes the ``K+1``-th key slot: a ray the point cloud cannot explain scores it
        highest and takes the background colour. Both fallbacks in
        :func:`~utils.geometry.ray_sphere_intersection` (missed sphere, exit point behind the
        camera) are deliberate -- the slot must exist for every ray, including degenerate ones.
        """
        if sphere_radius is None:
            sphere_radius = self.bkg_sphere_radius
        if sphere_center is None:
            sphere_center = self.bkg_sphere_center
        return ray_sphere_intersection(rays_o, rays_d, sphere_center, sphere_radius)

    def compute_distance_scale(self, pd: Tensor, rays: int) -> Tensor:
        """Multiplicative prior that damps far slots' scores, per ray.

        Reachable: the two editing configs set ``scale_scores_by_dist: true``. Slots are ranked by
        along-ray depth *within their own ray*, normalised to ``w`` in ``[0, 1]``, and scaled by
        ``exp(far + (near - far) * (1 - w**power))`` -- flat near the surface, dropping off sharply
        beyond it. Rays whose depth spread is below ``attn_dist_min_spread`` are exempted entirely,
        which keeps the prior from amplifying noise where every candidate is equidistant.

        Args:
            pd: ``(N, H, W, K+1, 1)`` unsigned along-ray depths, background slot included.
            rays: ``N * H * W``, the row count of the score matrix this multiplies.

        Returns:
            ``(rays, K+1)``.
        """
        strategy = self.args.get("attn_score_dist_strategy", "gamma")
        if strategy != "gamma":
            raise NotImplementedError(
                f"attn_score_dist_strategy={strategy!r} is not implemented; every shipped config "
                "that enables the distance prior uses 'gamma'"
            )

        eps = 1e-6
        mins = pd.amin(dim=-2, keepdim=True)
        maxs = pd.amax(dim=-2, keepdim=True)
        w = ((pd - mins) / (maxs - mins + eps)).squeeze(-1)

        power = self.args.get("attn_dist_power", 2.0)
        near_exp = self.args.get("attn_dist_near_exp", 1.0)
        far_exp = self.args.get("attn_dist_far_exp", -0.2)
        dist_scale = torch.exp(far_exp + (near_exp - far_exp) * (1.0 - w.pow(power)))

        min_spread = self.args.get("attn_dist_min_spread", None)
        if min_spread is not None and min_spread > 0:
            flat = (maxs - mins) <= min_spread
            if flat.any():
                dist_scale = torch.where(
                    flat.squeeze(-1).expand_as(w), torch.ones_like(w), dist_scale
                )
        return dist_scale.reshape(rays, -1)


    def forward(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        points: Tensor,
        point_features: Tensor,
        select_k_ind: Tensor,
        *,
        append_bkg_points_feats: Tensor,
        points_scaler: Tensor | None = None,
        step: int = -1,
        scores_only: bool = False,
    ) -> AttentionOutput | Tensor:
        """Score and value every (ray, candidate) pair.

        Args:
            rays_o: ``(N, 3)`` ray origins, one per view. ``k_type: 13`` rectifies points into each
                ray's frame from a per-view origin, so a per-ray ``(N, H, W, 3)`` origin is not
                accepted here.
            rays_d: ``(N, H, W, 3)`` unit ray directions.
            points: ``(M, 3)`` point cloud, in the frame the attention scores in.
            point_features: ``(M, D)`` per-point features, or ``(M, num_bases, D)`` under
                ``use_sh``.
            select_k_ind: ``(N, H, W, K)`` indices into ``points``, one set per ray.
            append_bkg_points_feats: ``(1, D)`` learned background feature. Required: every shipped
                config sets ``append_bkg_points: true``, so the background slot always exists.
            points_scaler: ``(M, 1)`` per-point radial scale on the pair geometry, or ``None``. The
                background slot is always scaled by 1.
            step: Training step, used only by the spherical-harmonic band schedule.
            scores_only: Return just the activated scores and skip the value branch. The
                surface-point path uses this.

        Returns:
            An :class:`AttentionOutput`, or the ``(N*H*W, K+1)`` score tensor if ``scores_only``.
        """
        if append_bkg_points_feats is None:
            raise ValueError(
                "append_bkg_points_feats is required: append_bkg_points is true in every shipped "
                "config, so the background slot has no feature without it"
            )
        if append_bkg_points_feats.dim() != 2:
            raise NotImplementedError(
                "append_bkg_points_feats must be (1, D); the per-ray cubemap background is not "
                f"ported, got {tuple(append_bkg_points_feats.shape)}"
            )

        selected_points = points[select_k_ind]
        sphere_intersection = self.get_bkg_sphere_intersection(rays_o, rays_d).unsqueeze(-2)
        selected_points = torch.cat([selected_points, sphere_intersection], dim=-2)
        n, h, w, slots, _ = selected_points.shape
        rays = n * h * w

        geom = ray_point_geometry(rays_o, rays_d, selected_points)
        pd, d2r, vec_pd, vec_d2r = geom.pd, geom.d2r, geom.vec_pd, geom.vec_d2r

        if points_scaler is not None:
            selected_scaler = points_scaler[select_k_ind]
            bkg_scaler = torch.ones(n, h, w, 1, 1, dtype=selected_scaler.dtype,
                                    device=selected_scaler.device)
            selected_scaler = torch.cat([selected_scaler, bkg_scaler], dim=-2)
            d2r = d2r * selected_scaler
            vec_d2r = vec_d2r * selected_scaler
            if not self.args.scale_d2r_only:
                vec_pd = vec_pd * selected_scaler
                pd = pd * selected_scaler

        key = key_features(
            self.args.k_type, rays_o=rays_o, rays_d=rays_d, selected_points=selected_points,
            vec_pd=vec_pd, vec_d2r=vec_d2r, pd=pd, d2r=d2r,
        )
        key = self.model_k(key.flatten(0, -2))
        query = query_features(self.args.q_type, rays_o=rays_o, rays_d=rays_d)
        query = self.model_q(query.flatten(0, -2))
        _, kq_dim = query.shape

        # downstream is stable there. Every shipped config sets use_amp: true, amp_dtype: float16,
        with torch.autocast(device_type="cuda", enabled=False):
            scores = torch.einsum(
                "ij,ikj->ik", query.float(), key.reshape(rays, -1, kq_dim).float()
            ) / math.sqrt(kq_dim)

        scaled_scores = scores * self.score_scale * self.args.score_scale_factor
        if self.args.get("scale_scores_by_dist", False):
            scaled_scores = scaled_scores * self.compute_distance_scale(pd, rays)
        scores = self.score_act(scaled_scores)

        if scores_only:
            return scores

        values = self._values(
            step, rays_o, point_features, select_k_ind, selected_points,
            vec_pd, vec_d2r, append_bkg_points_feats, (n, h, w, slots),
        )
        return AttentionOutput(
            scores=scores,
            values=values,
            select_k_ind=select_k_ind,
            sphere_intersection=sphere_intersection,
            pd=pd,
            d2r=d2r,
        )

    def _values(
        self,
        step: int,
        rays_o: Tensor,
        point_features: Tensor,
        select_k_ind: Tensor,
        selected_points: Tensor,
        vec_pd: Tensor,
        vec_d2r: Tensor,
        append_bkg_points_feats: Tensor,
        shape: tuple[int, int, int, int],
    ) -> Tensor:
        """One value vector per (ray, slot), from whichever of the two released value paths
        applies."""
        n, h, w, _slots = shape

        if self.args.use_pc_feats_directly:
            if not self.args.use_sh:
                raise NotImplementedError(
                    "use_pc_feats_directly requires use_sh in this release; the raw-feature "
                    "variant is unreachable from every shipped config"
                )
            num_bases = (self.args.sh_degree + 1) ** 2
            feat_dim = point_features.shape[-1]
            rays_to_points = selected_points - rays_o.reshape(n, 1, 1, 1, 3)
            coeffs = point_features[select_k_ind].reshape(n, h, w, -1, num_bases, feat_dim)
            bkg = append_bkg_points_feats.squeeze(0).expand(n, h, w, 1, -1)
            coeffs = torch.cat([coeffs, bkg.reshape(n, h, w, 1, num_bases, feat_dim)], dim=-3)
            degree = active_sh_degree(int(step), self.args.sh_degree_interval, self.args.sh_degree)
            values = spherical_harmonics(degree, rays_to_points, coeffs)
            return make_activation(self.args.pc_feats_act)(values)

        value_feats = value_features(
            self.args.v_type, vec_pd=vec_pd.reshape(-1, 3), vec_d2r=vec_d2r.reshape(-1, 3)
        )
        selected_feats = point_features[select_k_ind]
        bkg = append_bkg_points_feats.squeeze(0).expand(n, h, w, 1, -1)
        selected_feats = torch.cat([selected_feats, bkg], dim=-2)
        selected_feats = selected_feats.reshape(-1, selected_feats.shape[-1])

        values = self.model_v[0](value_feats)
        values = torch.cat([values, selected_feats], dim=-1)
        if self.args.network_v:
            values = self.model_v[1](values)
        return values


def _reject_unreachable(args: Any) -> None:
    """Fail at construction for any config that selects a path this release does not implement."""
    if not args.append_bkg_points:
        raise NotImplementedError(
            "append_bkg_points: false is not implemented; every shipped config appends the "
            "ray/background-sphere intersection as the last attention slot"
        )
    unreachable = {
        "shared_network": "a single trunk shared by k, q and v",
        "use_self_attn": "self-attention over the keys instead of a query",
        "prop_network_k": "the proposal network",
        "prop_network_q": "the proposal network",
        "select_v_scores": "top-k reselection before the value pass",
        "use_score_scaler": "the last key channel used as a per-pair score multiplier",
        "scale_bkg_token_by_num_bkg_points": "the learned background-token count scaler",
        "add_random_shift_to_vec_pd": "the random along-ray key jitter",
        "bkg_exp_scaler": "a learned scaler on the background slot inside the softmax",
        "predict_hit": "the auxiliary hit predictor",
    }
    for key, description in unreachable.items():
        if args.get(key, False):
            raise NotImplementedError(
                f"models.attn.{key} selects {description}, which is not ported: it is off in all "
                "nine released configs"
            )
    if not (args.network_k and args.network_q and args.network_v):
        raise NotImplementedError(
            "network_k / network_q / network_v are true in every shipped config; the network-free "
            "variants average the raw encoding instead of scoring it"
        )
    if args.encode_v_config.otype == "SphericalHarmonics":
        raise NotImplementedError(
            "encode_v_config.otype 'SphericalHarmonics' selects the tinycudann SH value encoding, "
            "which no shipped config uses; models.attn.use_sh drives the released SH path instead"
        )
    factorized = args.get("factorized_score", None)
    if factorized is not None and factorized.get("use", False):
        raise NotImplementedError(
            "models.attn.factorized_score.use selects the factorized-table score path and its CUDA "
            "kernels, which are not part of this release"
        )


def attention_weights(scores: Tensor, temp: float = 1.0) -> Tensor:
    """Softmax the scores over the candidate slots.

    Args:
        scores: ``(N, H, W, K+1, 1)``, background slot last.
        temp: ``attn_act_temp``. 1.0 in every shipped config.

    The reduction is over ``dim=-2``, the slot axis, because the scores carry a trailing singleton
    channel so they broadcast against the value vectors without a reshape.

    Only the plain softmax reduction is implemented here: the alternative normalisations
    (``minmax``, ``sum``, ``softpick``) and a learned scaler on the background slot's exponential
    (``bkg_exp_scaler``) are not reachable from any shipped config (``attn_act: softmax`` and
    ``bkg_exp_scaler: false`` in all nine) and were not ported. The scaler variant is not merely a
    special case of softmax either -- it divides by ``sum + eps`` where plain softmax does not --
    so folding the two together would not have been equivalent.
    """
    return torch.softmax(scores * temp, dim=-2)


def apply_influence_scores(
    scores: Tensor, selected_influ_scores: Tensor | None, fuse_type: str
) -> Tensor:
    """Fold each point's learned influence score into its attention score.

    The influence score is the per-point scalar the pruning schedule reads: a point whose influence
    stays low contributes nothing to any ray and is removed. Fusing it into the score before the
    softmax is what gives it that meaning.

    Args:
        scores: ``(N, H, W, K+1, 1)``, background slot last.
        selected_influ_scores: ``(N, H, W, K, 1)`` per-point influence, background slot **absent**.
            ``None`` leaves the scores untouched.
        fuse_type: ``influ_scores_fuse_type``. ``"add"`` in every shipped config; ``"multiply"`` is
            the ``configs/default.yml`` default and so stays implemented.

    The background slot is padded with the identity of whichever fusion is in use -- 0 for ``add``,
    1 for ``multiply`` -- so influence never moves mass between foreground and background.
    """
    if selected_influ_scores is None:
        return scores
    n, h, w = scores.shape[:3]
    pad_shape = (n, h, w, 1, 1)
    kwargs = {"dtype": selected_influ_scores.dtype, "device": selected_influ_scores.device}
    if fuse_type == "add":
        padded = torch.cat([selected_influ_scores, torch.zeros(pad_shape, **kwargs)], dim=-2)
        return scores + padded
    if fuse_type == "multiply":
        padded = torch.cat([selected_influ_scores, torch.ones(pad_shape, **kwargs)], dim=-2)
        return scores * padded
    raise ValueError(f"unknown influ_scores_fuse_type {fuse_type!r}; expected 'add' or 'multiply'")

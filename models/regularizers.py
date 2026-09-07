from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from models.features import ray_point_geometry
from models.surface import get_surface_points
from utils.geometry import normalize_vector

__all__ = [
    "REGULARIZER_NAMES",
    "SCHEDULE_TYPES",
    "compute_regularizers",
    "create_boundary_mask",
    "erode_mask_by_pixel_count",
    "gt_sp_loss",
    "point_on_ray_loss",
    "point_to_sp_loss",
    "scheduled_loss",
    "scheduled_weight",
]

#: Weight schedules within a loss's ``[start, stop]`` window.
SCHEDULE_TYPES: tuple[str, ...] = ("constant", "cosine", "cosine_decay", "cosine_warmup")

#: The regularizers this release assembles, in the order :func:`compute_regularizers` computes them.
#: ``cubemap_tv_loss``, ``opacity_sharpness_loss``, ``surface_planarity_loss``,
#: ``normal_consistency_loss``, ``depth_distortion_loss`` and a refiner band-limit term are not
#: included. Each belongs to a post-paper arm whose state this release does not build (a
#: cubemap background, the factorized-opacity tables, learnable per-point normals, the refiner), so
#: each would have been a permanently-zero entry referring to attributes that do not exist.
REGULARIZER_NAMES: tuple[str, ...] = (
    "point_on_ray_loss",
    "point_to_sp_loss",
    "if_score_var_loss",
    "alpha_loss",
    "influ_large_loss",
    "gt_pcd_loss",
    "gt_sp_loss",
)


def create_boundary_mask(foreground_mask: Any, boundary_width: int = 1) -> Any:
    """Zero out the band where foreground meets background, one at every other pixel.

    The depth supervision is least trustworthy exactly at a silhouette: the ground-truth depth
    jumps by the object's whole extent across one pixel, and any sub-pixel misalignment between the
    external reconstruction and this render turns into a metres-large residual that dominates the
    mean. Excluding the band is cheaper and more stable than trying to weight it.

    A pixel is on the boundary when dilating and eroding the foreground disagree there, which is a
    morphological gradient computed with two max-pools.

    Args:
        foreground_mask: ``(H, W)``, ``(B, H, W)`` or ``(B, 1, H, W)``, foreground 1 / background 0.
            Accepts a numpy array (returned as numpy) or a tensor.
        boundary_width: Half-width in pixels; the structuring element is ``2 * width + 1`` square.

    Returns:
        The same shape, dtype and container as the input: 0 on the boundary, 1 elsewhere.
    """
    is_numpy = isinstance(foreground_mask, np.ndarray)
    if is_numpy:
        foreground_mask = torch.from_numpy(foreground_mask)

    original_shape = foreground_mask.shape
    original_dtype = foreground_mask.dtype
    device = foreground_mask.device

    mask = foreground_mask.float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)

    kernel_size = 2 * boundary_width + 1
    _kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)

    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=boundary_width)
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=boundary_width)

    boundary = (dilated - eroded) > 0
    boundary_mask = (~boundary).float()

    if len(original_shape) == 2:
        boundary_mask = boundary_mask.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        boundary_mask = boundary_mask.squeeze(1)

    boundary_mask = boundary_mask.to(original_dtype)
    if is_numpy:
        boundary_mask = boundary_mask.numpy()
    return boundary_mask


def erode_mask_by_pixel_count(
    mask: Tensor,
    pixels_to_erode: int,
    kernel_size: int = 3,
    max_iterations: int | None = None,
) -> Tensor:
    """Shrink a boolean mask so its boundary retreats ``pixels_to_erode`` pixels.

    Distinct from :func:`create_boundary_mask`: that one carves out a symmetric band around the
    silhouette, this one only removes from the inside. The self-supervised losses use it because
    their mask comes from the *model's own* background attention, which is soft and unreliable near
    the edge; the ground-truth-depth loss uses the boundary version, since its mask is the GT alpha.

    Erosion is iterated with a small kernel rather than done once with a large one, so the shrink is
    measured in pixels of boundary retreat regardless of ``kernel_size``.

    Args:
        mask: Boolean, ``(N, H, W)`` or ``(H, W)``.
        pixels_to_erode: Radius to shrink by. ``<= 0`` returns a copy untouched.
        kernel_size: Window per erosion step.
        max_iterations: Ignored; kept because callers pass it.

    Returns:
        A boolean mask of the input's shape.
    """
    if pixels_to_erode <= 0:
        return mask.clone()

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    padding = kernel_size // 2
    result = mask.float().unsqueeze(1)

    for _ in range(int(pixels_to_erode)):
        inv_mask = 1.0 - result
        next_inv = F.max_pool2d(inv_mask, kernel_size=kernel_size, stride=1, padding=padding)
        result = 1.0 - next_inv
        if result.sum() == 0:
            break

    result = result.squeeze(1).bool()
    if squeeze_output:
        result = result.squeeze(0)
    return result


def scheduled_weight(
    base_weight: float,
    step: int,
    start: float,
    stop: float,
    schedule_type: str = "constant",
) -> float:
    """The weight a regularizer carries at ``step``.

    A geometry prior that is right at convergence is wrong at initialisation, when the point cloud
    is a uniform cube and "pull the points onto the surface the attention believes in" means
    "collapse the cube onto noise". Hence the window: every prior is off until the render is
    roughly right, and some ramp rather than switch on.

    Args:
        base_weight: The configured weight. ``<= 0`` short-circuits to 0.0.
        step: Current training step.
        start: First step the loss is active (inclusive).
        stop: Last step the loss is active (inclusive).
        schedule_type: One of :data:`SCHEDULE_TYPES`. An unrecognised value falls through to
            ``constant``, kept so a typo'd schedule trains the way the archived run did rather
            than raising mid-run.

    Returns:
        The weight as a plain float, 0.0 outside ``[start, stop]``.
    """
    if base_weight <= 0:
        return 0.0
    if step < start or step > stop:
        return 0.0
    span = max(float(stop - start), 1.0)
    progress = (step - start) / span
    if schedule_type in ("cosine", "cosine_decay"):
        ramp = 0.5 * (1.0 + torch.cos(torch.tensor(progress) * torch.pi))
        return float(base_weight * ramp.item())
    if schedule_type == "cosine_warmup":
        ramp = 0.5 * (1.0 - torch.cos(torch.tensor(progress) * torch.pi))
        return float(base_weight * ramp.item())
    return float(base_weight)


def scheduled_loss(
    loss_name: str,
    cur_step: int,
    loss_fn: Callable[[], Tensor],
    *,
    training: Any,
    device: torch.device | str,
    extra_condition: bool = True,
) -> tuple[Tensor, float]:
    """Evaluate a regularizer only if it is switched on at this step.

    ``loss_fn`` is a thunk, not a tensor, because every one of these costs a gather over ``K``
    candidates per ray and most are permanently off. Reading the weight first is what makes a config
    with six disabled regularizers cost nothing.

    Args:
        loss_name: Config key stem, e.g. ``point_on_ray_loss``. The window is read from
            ``<name>_weight`` / ``_start`` / ``_stop`` / ``_schedule``, falling back to
            ``regularizer_schedule`` for the last.
        cur_step: Current training step.
        loss_fn: Called only when the weight is positive and ``extra_condition`` holds.
        training: The ``training`` config node.
        device: Where to place the empty-sum placeholder.
        extra_condition: A second gate for state that may not exist yet (see the ``pruned_points``
            note in :func:`compute_regularizers`).

    Returns:
        ``(loss, weight)``. When the loss is off, ``loss`` is ``torch.zeros(0).sum()`` -- a 0.0
        scalar with no graph, so multiplying by the weight and adding it in is a no-op.
    """
    schedule = training.get(
        f"{loss_name}_schedule", training.get("regularizer_schedule", "constant")
    )
    weight = scheduled_weight(
        training.get(f"{loss_name}_weight", 0.0),
        cur_step,
        training.get(f"{loss_name}_start", -1e9),
        training.get(f"{loss_name}_stop", 1e9),
        schedule,
    )
    if weight > 0 and extra_condition:
        loss = loss_fn()
    else:
        loss = torch.zeros(0, device=device).sum()
    return loss, weight


def _split_attention(attn: Tensor) -> tuple[Tensor, Tensor]:
    """Separate the foreground candidates from the appended background slot.

    Every shipped config runs ``models.attn.append_bkg_points: true`` with
    ``geoms.points.use_bkg_points: false`` and ``geoms.no_additional_bkg: false``, which fixes the
    slot layout: the last slot is the ray/sphere background token, the first ``K`` are the selected
    points. Other configurations would fold in a separate background *point cloud* or a
    ``bkg_feats`` token with a different slot layout; none is reachable, and each would compute a
    different ``bkg_attn``.

    One consequence worth stating, because it looks like a bug and is not: the
    ``point_{on_ray,to_sp}_loss_norm_topk_attn`` options (``true`` in every shipped config) were
    only ever applied inside the *unreachable* ``bkg_feats`` branch. On this path the top-k weights
    are used un-renormalised, so those two config keys have no effect. Preserved.
    """
    return attn[..., :-1, :], attn[..., -1:, :]


def point_to_sp_loss(
    args: Any,
    points: Tensor,
    rays_o: Tensor,
    rays_d: Tensor,
    select_k_ind: Tensor,
    attn: Tensor,
    *,
    sphere_intersection: Tensor,
    coord_scale: float,
    gt_mask: Tensor | None = None,
    step: int = -1,
) -> Tensor:
    """Pull each ray's candidates toward the point the attention already fused from them.

    Self-supervised and deliberately circular: the target is the model's own attention-weighted
    mean, so the loss says "agree with yourself" -- it sharpens a distribution that is already
    roughly right rather than telling it where to be. That is why it is gated behind a start step.

    The fused point comes back from :func:`~models.surface.get_surface_points` in scene units, and
    ``points`` are in the renderer's scaled frame, so it is multiplied *up* by ``coord_scale`` here.
    :func:`gt_sp_loss` scales the other direction; see the module docstring.

    Args:
        args: The resolved config tree.
        points: ``(M, 3)`` point cloud in the scaled frame.
        rays_o: ``(N, 3)`` or ``(N, H, W, 3)`` ray origins.
        rays_d: ``(N, H, W, 3)`` unit ray directions.
        select_k_ind: ``(N, H, W, K)`` candidate indices.
        attn: ``(N, H, W, K+1, 1)`` attention weights, background slot last.
        sphere_intersection: ``(N, H, W, 1, 3)`` background-slot positions.
        coord_scale: The renderer's coordinate scale.
        gt_mask: ``(N, H, W, 1)`` ground-truth alpha, needed when masking by GT.
        step: Training step, for the periodic diagnostic print.

    Returns:
        Scalar. NaN when the mask empties -- see the module docstring.
    """
    use_bkg_points = False
    _topk_attn, bkg_attn = _split_attention(attn)

    selected_points = points[select_k_ind]
    if args.models.point_to_sp_loss_detach_attn:
        _topk_attn = _topk_attn.detach()

    surface = get_surface_points(
        points,
        select_k_ind,
        rays_o,
        rays_d,
        attn,
        args.models.point_to_sp_loss_fuse_type,
        sphere_intersection=sphere_intersection,
        coord_scale=coord_scale,
        divide_by_coord_scale=args.texture.divide_by_coord_scale,
    )
    sp = surface.unsqueeze(-2)
    if args.texture.divide_by_coord_scale:
        sp = sp * coord_scale
    if args.models.point_to_sp_loss_detach_sp:
        sp = sp.detach()

    distance_to_sp = torch.norm(selected_points - sp, dim=-1)
    if step >= 0 and step % 200 == 0:
        print(
            " distance_to_sp:", step, distance_to_sp.shape, distance_to_sp.min().item(),
            distance_to_sp.max().item(), distance_to_sp.mean().item(), distance_to_sp.std().item(),
        )

    if args.models.point_to_sp_loss_mask_thresh > 0:
        if args.models.point_to_sp_loss_use_gt_mask or use_bkg_points:
            mask = gt_mask.squeeze(-1) > 0.5
        else:
            mask = bkg_attn.squeeze(-1).squeeze(-1) < args.models.point_to_sp_loss_mask_thresh
        if args.models.point_to_sp_loss_erose_mask:
            mask = erode_mask_by_pixel_count(
                mask,
                pixels_to_erode=args.models.point_to_sp_loss_erose_mask_iters,
                kernel_size=5,
            )
        distance_to_sp = distance_to_sp[mask.detach()]
    if args.models.point_to_sp_loss_distance_thresh > 0 and not use_bkg_points:
        distance_to_sp = distance_to_sp[
            distance_to_sp < args.models.point_to_sp_loss_distance_thresh
        ]
    return distance_to_sp.mean()


def gt_sp_loss(
    args: Any,
    points: Tensor,
    rays_o: Tensor,
    rays_d: Tensor,
    select_k_ind: Tensor,
    attn: Tensor,
    gt_surface_points: Tensor,
    *,
    sphere_intersection: Tensor,
    coord_scale: float,
    gt_mask: Tensor | None = None,
    step: int = -1,
) -> Tensor:
    """Match the fused surface point to an externally reconstructed one.

    The only regularizer here with supervision from outside the model: the dataset unprojects a
    2D-Gaussian-Splatting depth map into world-space surface points and this drags the attention's
    depth expectation onto them. It is what makes the editing configs' geometry metric rather than
    merely self-consistent.

    Three things are hardcoded rather than configurable, all deliberately:

    * the fusion is ``position-sphere``, not ``point_to_sp_loss_fuse_type``. The GT point exists for
      background rays too, so the background slot's mass has to land somewhere real (the sphere)
      instead of being dropped -- otherwise a background ray's "surface point" drifts toward
      whichever stray foreground candidates it happened to select and the loss chases noise.
    * the metric is smooth L1, not the L2 of :func:`point_to_sp_loss`. An external depth map has
      outliers; a squared penalty would let a handful of bad pixels dominate.
    * the *target* is divided down by ``coord_scale`` while the prediction is left in scene units,
      the opposite of :func:`point_to_sp_loss`. Both are correct; see the module docstring.

    Args:
        args: The resolved config tree.
        points: ``(M, 3)`` point cloud in the scaled frame.
        rays_o: ``(N, 3)`` or ``(N, H, W, 3)`` ray origins.
        rays_d: ``(N, H, W, 3)`` unit ray directions.
        select_k_ind: ``(N, H, W, K)`` candidate indices.
        attn: ``(N, H, W, K+1, 1)`` attention weights, background slot last.
        gt_surface_points: ``(N, H, W, 3)`` external targets, in the dataset's own frame.
        sphere_intersection: ``(N, H, W, 1, 3)`` background-slot positions.
        coord_scale: The renderer's coordinate scale.
        gt_mask: ``(N, H, W, 1)`` ground-truth alpha, for the boundary exclusion.
        step: Training step, for the periodic diagnostic prints.

    Returns:
        Scalar, **NaN when the boundary mask leaves no pixels** -- a small enough patch that is
        entirely silhouette empties it. The dispatch's ``nan_to_num`` sweep converts that to exactly
        0.0 without a warning, which is why an all-boundary batch shows up as a step that trained
        the render but not the geometry, and shows up nowhere in the logs.
    """
    surface_points = get_surface_points(
        points,
        select_k_ind,
        rays_o,
        rays_d,
        attn,
        "position-sphere",
        sphere_intersection=sphere_intersection,
        coord_scale=coord_scale,
        divide_by_coord_scale=args.texture.divide_by_coord_scale,
    ).unsqueeze(-2)
    if args.texture.divide_by_coord_scale:
        gt_surface_points = gt_surface_points / coord_scale

    distance_to_sp = F.smooth_l1_loss(
        surface_points.squeeze(-2), gt_surface_points, reduction="none"
    ).mean(dim=-1)

    if args.training.gt_sp_loss_exclude_boundary and gt_mask is not None:
        boundary_width = args.training.gt_sp_loss_boundary_width
        boundary_mask = create_boundary_mask(gt_mask.squeeze(-1), boundary_width=boundary_width)
        boundary_mask = boundary_mask.view_as(distance_to_sp)
        if step >= 0 and step % 200 == 0:
            num_boundary_pixels = (boundary_mask < 0.5).sum().item()
            num_total_pixels = boundary_mask.numel()
            print(
                " gt_sp_loss boundary mask: {}/{} pixels excluded (width={})".format(
                    num_boundary_pixels, num_total_pixels, boundary_width
                )
            )
        distance_to_sp = distance_to_sp[boundary_mask > 0.5]

    if step >= 0 and step % 200 == 0:
        print(
            " distance_to_sp:", step, distance_to_sp.shape, distance_to_sp.min().item(),
            distance_to_sp.max().item(), distance_to_sp.mean().item(), distance_to_sp.std().item(),
        )
    return distance_to_sp.mean()


def point_on_ray_loss(
    args: Any,
    points: Tensor,
    rays_o: Tensor,
    rays_d: Tensor,
    select_k_ind: Tensor,
    attn: Tensor,
    *,
    gt_mask: Tensor | None = None,
) -> Tensor:
    """Push the attention's fused point onto the ray it was fused for.

    The attention blends ``K`` points that are scattered *around* the ray, so their weighted mean
    generally sits off it. The distance from that mean to the ray measures how much the attention is
    hedging laterally: drive it to zero and the ``K`` candidates have to line up along the ray,
    which is what turns a fog into a surface.

    Two metrics are implemented:

    * ``distance2`` -- the perpendicular distance from the fused point to the ray, in scene units.
      The only one any shipped config uses (``dress_papr``, weight 5e-3 from step 25k).
    * ``angle`` -- ``1 - cos`` between the ray and the direction to the fused point. Scale-free,
      so a distant point is penalised no harder than a near one for the same angular error.

    ``distance``, ``distance3``, ``vector`` and ``vector2`` are dropped: two of them read
    ``self.proximity_attn.pd`` / ``.vec_d2r``, attributes a renderer would have to stash on the
    module during the forward pass -- precisely the aliasing hazard this release removed.

    Note what this actually trains in the one config that enables it. ``dress_papr`` sets
    ``point_on_ray_loss_detach_points: true``, so ``selected_points`` carries no gradient and the
    only path back is through ``topk_attn``: the prior sharpens the *attention* until its
    expectation lands on the ray, rather than moving the points to meet it. With
    ``detach_points: false`` it does both.

    Args:
        args: The resolved config tree.
        points: ``(M, 3)`` point cloud in the scaled frame.
        rays_o: ``(N, 3)`` or ``(N, H, W, 3)`` ray origins.
        rays_d: ``(N, H, W, 3)`` unit ray directions.
        select_k_ind: ``(N, H, W, K)`` candidate indices.
        attn: ``(N, H, W, K+1, 1)`` attention weights, background slot last.
        gt_mask: ``(N, H, W, 1)`` ground-truth alpha, needed when masking by GT.

    Returns:
        Scalar.
    """
    use_bkg_points = False
    topk_attn, bkg_attn = _split_attention(attn)

    selected_points = points[select_k_ind]
    if args.models.point_on_ray_loss_detach_points:
        selected_points = selected_points.detach()

    loss_type = args.models.point_on_ray_loss_type
    if loss_type == "distance2":
        surface_points = torch.sum(selected_points * topk_attn, dim=3, keepdim=True)
        geometry = ray_point_geometry(rays_o, rays_d, surface_points)
        metric = geometry.d2r.squeeze(-2)
    elif loss_type == "angle":
        surface_points = torch.sum(selected_points * topk_attn, dim=3)
        rays = normalize_vector(surface_points - rays_o.reshape(rays_o.shape[0], 1, 1, 3))
        metric = torch.sum(rays * normalize_vector(rays_d), dim=-1, keepdim=True)
        metric = 1 - metric.clamp(0, 1)
    else:
        raise ValueError(
            f"unsupported models.point_on_ray_loss_type {loss_type!r}; this release implements "
            "'distance2' and 'angle'"
        )

    if args.models.point_on_ray_loss_mask_thresh > 0:
        if args.models.point_on_ray_loss_use_gt_mask or use_bkg_points:
            mask = gt_mask.squeeze(-1) > 0.5
        else:
            mask = bkg_attn.squeeze(-1).squeeze(-1) < args.models.point_on_ray_loss_mask_thresh
        if args.models.point_on_ray_loss_erose_mask:
            mask = erode_mask_by_pixel_count(
                mask,
                pixels_to_erode=args.models.point_on_ray_loss_erose_mask_iters,
                kernel_size=5,
            )
        metric = metric[mask.detach()]
    return metric.mean()


def compute_regularizers(
    args: Any,
    *,
    step: int,
    device: torch.device | str,
    points: Tensor,
    rays_o: Tensor,
    rays_d: Tensor,
    select_k_ind: Tensor,
    attn: Tensor,
    bkg_attn: Tensor,
    mask: Tensor,
    sphere_intersection: Tensor,
    coord_scale: float,
    points_influ_scores: Tensor | None = None,
    gt_points: Tensor | None = None,
    gt_surface_points: Tensor | None = None,
    pruned_points: bool = False,
) -> dict[str, tuple[Tensor, float]]:
    """Evaluate every regularizer for this step and return ``{name: (loss, weight)}``.

    The caller multiplies and sums; keeping the pair separate is what lets the training log show
    each term's raw magnitude next to its schedule, which is the only way to tell a term that is
    off from a term that has converged.

    ``gt_sp_loss`` is gated on ``pruned_points``, and that gate is asymmetric in a way that changes
    the objective. ``model.pruned_points`` first flips true at the *first prune* on a from-scratch
    run (``train.py``'s prune block), but is set immediately when a run starts from ``--load_path``.
    So a from-scratch run optimises without depth supervision for its first ``prune_start`` steps
    while a fine-tune has it from step 0 -- two different objectives from the same config. This is
    how the released checkpoints were trained; preserved deliberately.

    Args:
        args: The resolved config tree.
        step: Current training step.
        device: Where the zero placeholders live.
        points: ``(M, 3)`` point cloud in the scaled frame.
        rays_o: ``(N, 3)`` or ``(N, H, W, 3)`` ray origins.
        rays_d: ``(N, H, W, 3)`` unit ray directions.
        select_k_ind: ``(N, H, W, K)`` candidate indices, passed in rather than read off a module.
        attn: ``(N, H, W, K+1, 1)`` attention weights, background slot last.
        bkg_attn: ``(N, H, W, 1)`` the background slot's weight, as the renderer split it.
            The split is ``attn[..., -1, :]`` -- an index, not a slice -- so the slot axis
            is gone and this has one fewer dimension than ``attn``. Its only consumer,
            ``bkg_loss``, calls a bare ``.squeeze()`` and is insensitive to the rank.
        mask: ``(N, H, W, 1)`` ground-truth alpha for this patch.
        sphere_intersection: ``(N, H, W, 1, 3)`` background-slot positions.
        coord_scale: The renderer's coordinate scale.
        points_influ_scores: ``(M, 1)`` learned per-point influence, or ``None``.
        gt_points: ``(P, 3)`` reference cloud for the Chamfer term, or ``None``.
        gt_surface_points: ``(N, H, W, 3)`` external depth targets for this patch, or ``None``.
        pruned_points: Whether pruning has run (or the checkpoint was resumed). See above.

    Returns:
        ``{name: (loss, weight)}`` for :data:`REGULARIZER_NAMES`, every loss already passed through
        the ``nan_to_num`` sweep.
    """
    training = args.training

    def _scheduled(name: str, fn: Callable[[], Tensor], extra_condition: bool = True):
        return scheduled_loss(
            name, step, fn, training=training, device=device, extra_condition=extra_condition
        )

    on_ray_loss, on_ray_weight = _scheduled(
        "point_on_ray_loss",
        lambda: point_on_ray_loss(
            args, points, rays_o, rays_d, select_k_ind, attn, gt_mask=mask
        ),
    )

    to_sp_loss, to_sp_weight = _scheduled(
        "point_to_sp_loss",
        lambda: point_to_sp_loss(
            args, points, rays_o, rays_d, select_k_ind, attn,
            sphere_intersection=sphere_intersection, coord_scale=coord_scale,
            gt_mask=mask, step=step,
        ),
    )

    def _if_score_var() -> Tensor:
        scores = points_influ_scores[points_influ_scores > 1.01e-5]
        return scores.var(dim=0).mean()

    var_loss, var_weight = _scheduled(
        "if_score_var_loss", _if_score_var, points_influ_scores is not None
    )

    alpha_loss, alpha_weight = _scheduled(
        "alpha_loss",
        lambda: F.mse_loss(bkg_attn.squeeze(), (1 - mask).detach().squeeze()),
    )

    def _influ_large() -> Tensor:
        thresh = training.get("influ_large_loss_thresh", 1.0)
        return F.softplus(thresh - points_influ_scores).mean()

    influ_loss, influ_weight = _scheduled(
        "influ_large_loss", _influ_large, points_influ_scores is not None
    )

    def _gt_pcd() -> Tensor:
        from pytorch3d.loss import chamfer_distance

        return chamfer_distance(gt_points[None, ...], points[None, ...])[0].mean()

    pcd_loss, pcd_weight = _scheduled(
        "gt_pcd_loss", _gt_pcd, pruned_points and gt_points is not None
    )

    sp_loss, sp_weight = _scheduled(
        "gt_sp_loss",
        lambda: gt_sp_loss(
            args, points, rays_o, rays_d, select_k_ind, attn, gt_surface_points,
            sphere_intersection=sphere_intersection, coord_scale=coord_scale,
            gt_mask=mask, step=step,
        ),
        pruned_points and gt_surface_points is not None,
    )

    regularizers: dict[str, tuple[Tensor, float]] = {
        "point_on_ray_loss": (on_ray_loss, on_ray_weight),
        "point_to_sp_loss": (to_sp_loss, to_sp_weight),
        "if_score_var_loss": (var_loss, var_weight),
        "alpha_loss": (alpha_loss, alpha_weight),
        "influ_large_loss": (influ_loss, influ_weight),
        "gt_pcd_loss": (pcd_loss, pcd_weight),
        "gt_sp_loss": (sp_loss, sp_weight),
    }

    regularizers = {
        name: (
            torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.is_tensor(loss) else loss,
            weight,
        )
        for name, (loss, weight) in regularizers.items()
    }
    return regularizers

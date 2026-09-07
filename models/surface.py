from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from models.attention import ProximityAttention, apply_influence_scores, attention_weights
from models.features import ray_point_geometry

__all__ = [
    "FUSE_TYPES",
    "SurfaceSample",
    "get_surface_points",
    "get_surface_points_from_rays",
]

#: The fusions this release implements. ``projection`` (depth expectation along the ray, ignoring
#: the candidates' lateral offsets) and ``position-projection`` (``position`` then re-projected,
#: i.e. without the background slot) are not implemented: every released call site asks for one of
#: the three below.
FUSE_TYPES: tuple[str, ...] = ("position", "position-sphere", "position-sphere-projection")


class SurfaceSample(NamedTuple):
    """Everything :func:`get_surface_points_from_rays` computed on the way to the surface point."""

    #: ``(N, H, W, 3)`` surface points, divided by ``coord_scale``.
    surface_points: Tensor
    #: ``(N, H, W, K+1, 1)`` attention weights, background slot last.
    attn: Tensor
    #: ``(N, H, W, K, 3)`` the candidate points those weights belong to.
    selected_points: Tensor
    #: ``(N, H, W, K)`` indices of those candidates. Returned, never stashed on a module.
    select_k_ind: Tensor


def get_surface_points(
    points: Tensor,
    select_k_ind: Tensor,
    rays_o: Tensor,
    rays_d: Tensor,
    attn: Tensor,
    fuse_type: str,
    *,
    sphere_intersection: Tensor,
    coord_scale: float,
    divide_by_coord_scale: bool = True,
) -> Tensor:
    """Fuse a ray's attention weights into one surface point per ray.

    Args:
        points: ``(M, 3)`` point cloud, in the renderer's scaled frame.
        select_k_ind: ``(N, H, W, K)`` indices of each ray's candidates.
        rays_o: ``(N, 3)`` or ``(N, H, W, 3)`` ray origins.
        rays_d: ``(N, H, W, 3)`` unit ray directions.
        attn: ``(N, H, W, K+1, 1)`` attention weights with the background slot **last**, as
            :func:`~models.attention.attention_weights` produces them.
        fuse_type: One of :data:`FUSE_TYPES`.
        sphere_intersection: ``(N, H, W, 1, 3)`` background-slot positions, from
            :class:`~models.attention.AttentionOutput`. Passed in rather than read back off the
            attention module, so a tiled render cannot pair one tile's weights with another's
            geometry.
        coord_scale: The renderer's coordinate scale.
        divide_by_coord_scale: ``texture.divide_by_coord_scale``; true in every shipped config.

    Returns:
        ``(N, H, W, 3)``, except that ``position-sphere-projection`` additionally applies a
        ``squeeze(-2)`` -- see the note in the body.
    """
    if fuse_type not in FUSE_TYPES:
        raise ValueError(f"unknown surface fuse type {fuse_type!r}; expected one of {FUSE_TYPES}")

    expected_slots = select_k_ind.shape[-1] + 1
    if attn.shape[-2] != expected_slots:
        raise ValueError(
            f"attn has {attn.shape[-2]} slots but select_k_ind has {select_k_ind.shape[-1]} "
            f"candidates; the background slot must be present, giving {expected_slots}"
        )

    topk_attn = attn[..., :-1, :]
    bkg_attn = attn[..., -1:, :]

    if rays_o.ndim == 2:
        rays_o = rays_o.unsqueeze(1).unsqueeze(1)

    surface_points = torch.sum(points[select_k_ind] * topk_attn, dim=3)
    if fuse_type != "position":
        surface_points = surface_points + torch.sum(sphere_intersection * bkg_attn, dim=3)

    if fuse_type == "position-sphere-projection":
        along_ray = ray_point_geometry(rays_o, rays_d, surface_points.unsqueeze(-2)).vec_pd
        surface_points = (rays_o + along_ray.squeeze(-2)).squeeze(-2)

    if divide_by_coord_scale:
        surface_points = surface_points / coord_scale
    return surface_points


def get_surface_points_from_rays(
    attention: ProximityAttention,
    points: Tensor,
    point_features: Tensor,
    select_k_ind: Tensor,
    rays_o: Tensor,
    rays_d: Tensor,
    *,
    append_bkg_points_feats: Tensor,
    points_influ_scores: Tensor | None = None,
    points_scaler: Tensor | None = None,
    influ_fuse_type: str = "add",
    attn_temp: float = 1.0,
    fuse_type: str = "position",
    coord_scale: float = 1.0,
    divide_by_coord_scale: bool = True,
    step: int = -1,
) -> SurfaceSample:
    """Score a batch of rays and fuse their surface points, skipping the value branch.

    The scores are all this needs, so the attention runs with ``scores_only=True``: the value
    network, the SH expansion and the U-Net are never touched. That is what makes it affordable to
    call over every foreground pixel of a scene, which the editing stage does to build its surface
    sample set.

    The caller supplies ``select_k_ind`` rather than a selector, because choosing the candidates is
    the model's job (it owns the frustum-KNN index and its caches) and because the indices must be
    the *same* ones the attention scores -- passing them explicitly is what keeps that true.

    Args:
        attention: The module to score with.
        points: ``(M, 3)`` point cloud.
        point_features: ``(M, D)`` per-point features. Unused on this path, but the attention
            signature requires them.
        select_k_ind: ``(N, H, W, K)`` candidate indices.
        rays_o: ``(N, 3)`` or ``(N, H, W, 3)`` ray origins.
        rays_d: ``(N, H, W, 3)`` unit ray directions.
        append_bkg_points_feats: ``(1, D)`` learned background feature.
        points_influ_scores: ``(M, 1)`` learned per-point influence, or ``None``.
        points_scaler: ``(M, 1)`` per-point geometry scale, or ``None``.
        influ_fuse_type: ``influ_scores_fuse_type``.
        attn_temp: ``attn_act_temp``.
        fuse_type: One of :data:`FUSE_TYPES`.
        coord_scale: The renderer's coordinate scale.
        divide_by_coord_scale: ``texture.divide_by_coord_scale``.
        step: Training step. Unused with ``scores_only``, forwarded for symmetry.

    Returns:
        A :class:`SurfaceSample`.
    """
    selected_points = points[select_k_ind]

    scores = attention(
        rays_o,
        rays_d,
        points,
        point_features,
        select_k_ind,
        append_bkg_points_feats=append_bkg_points_feats,
        points_scaler=points_scaler,
        step=step,
        scores_only=True,
    )
    n, h, w = rays_d.shape[:3]
    scores = scores.reshape(n, h, w, -1, 1)

    selected_influ = None if points_influ_scores is None else points_influ_scores[select_k_ind]
    scores = apply_influence_scores(scores, selected_influ, influ_fuse_type)

    attn = attention_weights(scores, attn_temp)
    sphere_intersection = attention.get_bkg_sphere_intersection(rays_o, rays_d).unsqueeze(-2)
    surface_points = get_surface_points(
        points,
        select_k_ind,
        rays_o,
        rays_d,
        attn,
        fuse_type,
        sphere_intersection=sphere_intersection,
        coord_scale=coord_scale,
        divide_by_coord_scale=divide_by_coord_scale,
    )
    return SurfaceSample(surface_points, attn, selected_points, select_k_ind)

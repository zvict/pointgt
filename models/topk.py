from __future__ import annotations

from dataclasses import dataclass

import torch
from pytorch3d.ops import knn_points
from torch import Tensor

from utils.geometry import perspective_projection, project_points_to_image

__all__ = [
    "TopkSelection",
    "fused_projection",
    "perspective_projection",
    "select_topk_points",
]

#: Distance stand-in for a point outside the frustum: large enough to lose every comparison against
#: a real pixel distance. Note it overflows float16, so a
#: ``topk_dtype: float16`` config would compare infinities -- which still sorts correctly, since
#: every masked entry becomes the same infinity, but the value is chosen for float32.
_MASKED_DISTANCE = 1e10


def fused_projection(
    coords: Tensor,
    c2w: Tensor,
    fx: float | Tensor,
    fy: float | Tensor,
    cx: float | Tensor,
    cy: float | Tensor,
    width: int | float,
    *,
    vector: bool = False,
) -> tuple[Tensor, Tensor]:
    """World points to ``(pixel_uv, camera_space_points)`` in one pass.

    Thin alias for :func:`utils.geometry.project_points_to_image`, kept under this name because it
    is what the render path calls. The camera-space tensor is returned alongside the pixel
    coordinates because its ``z`` is the frustum test's depth and recomputing it would double the
    transform cost per step.

    The signature this replaces took ``H, W``; ``H`` was never read (only ``W``, by the horizontal
    flip in the projection). It is dropped here rather than kept as a decorative argument, which
    also means an un-updated eight-positional call site fails loudly instead of projecting with
    the height as the width.
    """
    return project_points_to_image(coords, c2w, fx, fy, cx, cy, width, vector=vector)


@dataclass(frozen=True)
class TopkSelection:
    """The per-ray shortlist and the pixel distance the background masks are cut from."""

    #: ``(N, H, W, K)`` int64 indices into the point cloud. May be an expanded view when the cloud
    #: is smaller than ``select_k``, so treat it as read-only.
    indices: Tensor
    #: ``(N, H, W)`` pixel distance to the closest projected point. Really computed on the
    #: ``d2r_z`` path, which forms the full pixel-distance matrix anyway and takes its row minimum
    #: for free. ``knn_frustum`` never forms that matrix -- the kNN kernel hands back neighbours,
    #: not a minimum over the whole cloud -- so that path returns zeros rather than paying
    #: ``N*H*W*P`` to fill a field nothing reads. See :func:`select_topk_points`.
    min_d2r: Tensor


def _pixel_distances(points_2d: Tensor, pix_coords: Tensor, dtype: torch.dtype) -> Tensor:
    """``(N, H, W, P)`` distance from every pixel centre to every projected point."""
    return torch.norm(
        points_2d.to(dtype)[:, None, None, :, :] - pix_coords.to(dtype)[..., None, :], dim=-1
    )


def select_topk_points(
    points_2d: Tensor,
    pix_coords: Tensor,
    z: Tensor,
    *,
    select_k: int,
    image_height: int,
    image_width: int,
    select_k_type: str = "knn_frustum",
    select_k_z_pow: float = 1.0,
    pixel_frustum_margin: float = 1.0,
    topk_dtype: torch.dtype = torch.float32,
    eps: float = 1e-6,
) -> TopkSelection:
    """Shortlist ``select_k`` candidate points per ray.

    Args:
        points_2d: ``(N, P, 2)`` or ``(P, 2)`` projected pixel coordinates of the whole point
            cloud, from :func:`fused_projection`. A shared ``(P, 2)`` is broadcast across views.
        pix_coords: ``(N, H, W, 2)`` or ``(H, W, 2)`` pixel centres of the rays being rendered.
            ``H, W`` are the *patch* size during training, which is why the frustum test needs the
            full image size separately.
        z: ``(N, P)`` or ``(N, 1, 1, P)`` camera-space depth of each point. Negative is in front of
            the camera (the Blender/OpenGL convention the projection assumes).
        select_k: Candidates per ray. ``< 0``, or ``>= P``, selects the whole cloud.
        image_height, image_width: Full image size, for the frustum test.
        select_k_type: ``"knn_frustum"`` or ``"d2r_z"``.
        select_k_z_pow: Exponent on the depth term of the ``d2r_z`` ranking. Ignored otherwise.
        pixel_frustum_margin: Slack in pixels on the image-bounds test, so a point that projects
            just outside the frame still lights the border rays it influences.
        topk_dtype: Precision of the pixel-distance comparison (``topk_dtype`` in the config).
        eps: Depth deadband on the in-front-of-camera test.

    Returns:
        A :class:`TopkSelection`.

    ``min_d2r`` is filled by ``d2r_z`` and zero on ``knn_frustum``; see :class:`TopkSelection` for
    why. Either way nothing in this release reads it. Its consumers were the
    ``mask_bkg_rays_thresh`` and ``training.bkg_reg_mask_thresh`` ray gates, both ``-1.0``
    (disabled) in every shipped config and therefore not ported -- the keys survive in the vendored
    defaults, but no code here looks at them. The field stays in the return type because it is a
    real quantity on one of the two shipped paths, and it is carried through
    :class:`~models.papr.RenderOutput` and :class:`~models.papr.EvalOutput` for the same reason.

    Raises:
        ValueError: for an unimplemented ``select_k_type`` or a shape that cannot be reconciled.
    """
    if select_k_type not in ("knn_frustum", "d2r_z"):
        raise ValueError(
            f"unsupported geoms.points.select_k_type {select_k_type!r}; this release implements "
            "'knn_frustum' and 'd2r_z'. The remaining selectors (d2r, d2r_filter, "
            "grid_knn, the traversal/streaming admission caches) were not ported: they need "
            "either compiled CUDA kernels this release does not ship, or a code path no released "
            "checkpoint uses."
        )

    if points_2d.ndim not in (2, 3) or points_2d.shape[-1] != 2:
        raise ValueError(f"points_2d must be (N, P, 2) or (P, 2), got {tuple(points_2d.shape)}")
    if pix_coords.ndim not in (3, 4) or pix_coords.shape[-1] != 2:
        raise ValueError(
            f"pix_coords must be (N, H, W, 2) or (H, W, 2), got {tuple(pix_coords.shape)}"
        )

    if z.ndim == 4:
        z_vals = z[:, 0, 0, :]
    elif z.ndim == 2:
        z_vals = z
    else:
        raise ValueError(f"z must be (N, P) or (N, 1, 1, P), got {tuple(z.shape)}")
    n_views = z_vals.shape[0]
    num_points = z_vals.shape[-1]
    height, width = pix_coords.shape[-3], pix_coords.shape[-2]
    device = points_2d.device

    if pix_coords.ndim == 3:
        pix_coords = pix_coords.unsqueeze(0).expand(n_views, -1, -1, -1)
    elif pix_coords.shape[0] == 1 and n_views != 1:
        pix_coords = pix_coords.expand(n_views, -1, -1, -1)
    elif pix_coords.shape[0] != n_views:
        raise ValueError(f"pix_coords has {pix_coords.shape[0]} views but z has {n_views}")

    min_d2r = torch.zeros(n_views, height, width, device=device)

    if select_k_type == "d2r_z":
        k = min(select_k, num_points) if select_k >= 0 else num_points
        pts = points_2d if points_2d.ndim == 3 else points_2d.unsqueeze(0).expand(n_views, -1, -1)
        distances = _pixel_distances(pts, pix_coords, topk_dtype)
        min_d2r = distances.min(dim=-1)[0]
        depth = z_vals[:, None, None, :].to(distances.dtype)
        ranking = distances * (1.0 - depth).pow(select_k_z_pow)
        _, indices = torch.topk(ranking, k, dim=-1, largest=False, sorted=False)
        return TopkSelection(indices=indices, min_d2r=min_d2r)

    if select_k >= num_points or select_k < 0:
        indices = torch.arange(num_points, device=device).expand(n_views, height, width, -1)
        return TopkSelection(indices=indices, min_d2r=min_d2r)

    if points_2d.ndim == 3:
        pts_2d = points_2d
    else:
        pts_2d = points_2d.unsqueeze(0).expand(n_views, -1, -1)

    in_front = z_vals < 0 - eps
    in_width = (pts_2d[..., 0] >= 0 - pixel_frustum_margin) & (
        pts_2d[..., 0] < image_width + pixel_frustum_margin
    )
    in_height = (pts_2d[..., 1] >= 0 - pixel_frustum_margin) & (
        pts_2d[..., 1] < image_height + pixel_frustum_margin
    )
    frustum_mask = in_front & in_width & in_height

    valid_per_view = frustum_mask.sum(dim=-1)
    max_valid = int(valid_per_view.max().item())

    if max_valid == 0:
        distances = _pixel_distances(pts_2d, pix_coords, topk_dtype)
        _, indices = torch.topk(distances, select_k, dim=-1, largest=False, sorted=False)
    elif max_valid < select_k:
        distances = _pixel_distances(pts_2d, pix_coords, topk_dtype)
        masked = distances.clone()
        masked[~frustum_mask[:, None, None, :].expand_as(distances)] = _MASKED_DISTANCE
        _, indices = torch.topk(masked, select_k, dim=-1, largest=False, sorted=False)
    else:
        sorted_indices = torch.argsort(frustum_mask.float(), dim=-1, descending=True)
        sorted_pts_2d = torch.gather(
            pts_2d, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 2)
        )[:, :max_valid, :]
        lengths = valid_per_view.clamp(min=1, max=max_valid)

        query_pts = pix_coords.reshape(n_views, height * width, 2)
        _, knn_idx, _ = knn_points(
            query_pts.to(topk_dtype),
            sorted_pts_2d.to(topk_dtype),
            lengths2=lengths,
            K=select_k,
            return_sorted=False,
        )
        sorted_indices_truncated = sorted_indices[:, :max_valid]
        indices = torch.gather(
            sorted_indices_truncated, 1, knn_idx.reshape(n_views, -1)
        ).reshape(n_views, height, width, select_k)

    return TopkSelection(indices=indices, min_d2r=min_d2r)

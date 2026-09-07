from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from edit.correspondence import transfer
from models.surface import get_surface_points
from uv.bbox import points_inside_bounding_box

__all__ = [
    "DEFAULT_FUSE_TYPE",
    "DEFAULT_TRANSFER_METHOD",
    "EditedFrame",
    "TextureAtlas",
    "check_edit_preconditions",
    "render_edited_view",
    "to_uint8",
]

#: The surface fusion the edit path uses. Alone among the three it re-projects the fused point onto
#: its ray, and the canonical transfer is defined for on-ray samples.
DEFAULT_FUSE_TYPE = "position-sphere-projection"

#: The paper's transfer. ``application/backend`` called it ``attn_translation``.
DEFAULT_TRANSFER_METHOD = "deformation_aware"


@runtime_checkable
class TextureAtlas(Protocol):
    """What this module needs from stage b: a way to read a texture map at 3D canonical points.

    Deliberately narrow. Chart assignment, the UV MLPs, the sigma field and the texture-map
    parameterisation are all stage b's business; the edit render only ever asks "what colour does
    *this* texture map have at *these* canonical points". Two texture maps are queried per frame
    with the same points -- the edited colours and the edit mask -- so the atlas is free to cache
    the UV prediction between the two calls.
    """

    #: Number of charts the atlas packs the surface into.
    num_charts: int

    def sample_texture(self, points: Tensor, texture_map: Tensor) -> Tensor:
        """Sample ``texture_map`` at ``(N, 3)`` canonical ``points``, returning ``(N, C)``."""


@dataclass(frozen=True)
class EditedFrame:
    """One rendered frame plus every intermediate the renderer produces.

    All images are float32 in ``[0, 1]`` with no batch axis, which is what an image writer wants;
    the tensors that keep a leading ``1`` are the ones whose rank the renderer's own consumers
    depend on.
    """

    #: ``(H, W, 3)`` the blended result -- what gets written to disk.
    final_rgb: Tensor
    #: ``(H, W, 3)`` the PAPR render of the deformed cloud, before any texture is applied.
    papr_rgb: Tensor
    #: ``(H, W, 3)`` the edited texture read through the atlas; zero outside the foreground.
    texture_rgb: Tensor
    #: ``(H, W, 1)`` the edit mask sampled at the surface; the blend's alpha.
    mask: Tensor
    #: ``(1, H, W, 3)`` fused surface points in *deformed* space, divided by ``coord_scale``.
    surface_points: Tensor
    #: ``(H, W)`` which pixels were looked up in the atlas at all.
    foreground: Tensor


def check_edit_preconditions(args: Any, *, checkpoint_points: int, deformed_points: int) -> None:
    """Refuse the two configurations under which this render is silently wrong.

    **Background points.** ``PAPR.get_points()`` *concatenates* a fixed background cloud after the
    real points when ``geoms.points.use_bkg_points`` or ``geoms.append_sphere_bkg_points`` is set,
    so ``select_k_ind`` then indexes an augmented tensor. The edit path hands the un-augmented
    deformed cloud to the surface fuse and to the canonical transfer, so every index at or past
    the real point count would address the wrong row -- of a *differently shaped* array in the
    transfer, which is the case that does not even raise. ``models.papr`` refuses both flags at
    construction today; checking here as well means this stays true if that ever loosens, and the
    message says why it matters *for this render* rather than in general.

    **Point count.** See :func:`edit.pcd.load_deformed_points`; repeated here so a caller that
    assembled its own deformed tensor is checked too.
    """
    point_opt = args.geoms.points
    if bool(point_opt.get("use_bkg_points", False)):
        raise ValueError(
            "geoms.points.use_bkg_points is true. The edit render cannot use a background point "
            "cloud: get_points() appends it to the real points, so select_k_ind indexes an "
            "augmented tensor, while the surface fuse and the canonical transfer here are given "
            "the un-augmented deformed cloud. The indices would silently address the wrong rows. "
            "Set geoms.points.use_bkg_points: false (both shipped editing configs do)."
        )
    if bool(args.geoms.get("append_sphere_bkg_points", False)):
        raise ValueError(
            "geoms.append_sphere_bkg_points is true, which appends sphere-initialised background "
            "points to the cloud and shifts the index space select_k_ind lives in. The edit "
            "render's surface fuse and canonical transfer index the un-augmented deformed cloud. "
            "Set geoms.append_sphere_bkg_points: false (both shipped editing configs do)."
        )
    if deformed_points != checkpoint_points:
        raise ValueError(
            f"the deformed cloud has {deformed_points} points, the checkpoint's has "
            f"{checkpoint_points}. Point i of a deformed frame must be point i of the "
            "checkpoint's cloud: pc_feats, points_influ_scores and points_scaler stay on the "
            "checkpoint's rows and are not reloaded per frame, and the canonical transfer pairs "
            "each deformed point with the canonical point of the same index."
        )


def _tile_bounds(extent: int, tile: int) -> Iterator[tuple[int, int]]:
    """``[start, end)`` spans covering ``extent``, the last one short."""
    if tile <= 0:
        raise ValueError(f"tile size must be positive, got {tile}")
    for start in range(0, extent, tile):
        yield start, min(start + tile, extent)


def _composite_background(model: Any, rgb: Tensor, attn: Tensor, bkg_color: Tensor) -> Tensor:
    """Blend the background colour in over the decoded RGB.

    Note this is *not* what ``test.py`` does: this composites the background colour over the
    render using the background slot's attention even when ``append_bkg_points`` is true, whereas
    ``test.py`` leaves the render alone because the background already reaches the image through
    that slot's own value features. Both are legitimate, and this is the behaviour kept here.
    """
    bkg_attn = attn[..., -1:, :].squeeze(-1)
    if model.args.models.attn.append_bkg_points:
        return rgb * (1 - bkg_attn) + bkg_color.squeeze() * bkg_attn
    if model.args.geoms.background.use_dumb_constant:
        bkg_attn = (bkg_attn * model.dumb_constant).clamp(0, 1)
    bkg_attn = model.bkg_attn_act(
        (bkg_attn + model.args.models.bkg_attn_shift) * model.args.models.bkg_attn_scale
    )
    return rgb * (1 - bkg_attn) + bkg_color.squeeze() * bkg_attn


@torch.no_grad()
def render_edited_view(
    model: Any,
    dataset: Any,
    *,
    atlas: TextureAtlas,
    texture_map: Tensor,
    mask_texture_map: Tensor,
    deformed_points: Tensor,
    canonical_points: Tensor,
    c2w: Tensor,
    max_height: int,
    max_width: int,
    bbox_hull: Any = None,
    bkg_color: Tensor | None = None,
    fuse_type: str = DEFAULT_FUSE_TYPE,
    transfer_method: str = DEFAULT_TRANSFER_METHOD,
    project_before_transfer: bool = False,
    step: int = -1,
) -> EditedFrame:
    """Render one frame of the edit.

    Args:
        model: A loaded :class:`~models.papr.PAPR`, already given its camera via ``set_camera``.
        dataset: The split the intrinsics and image size come from; only ``H``, ``W``,
            ``pix_coords`` and ``get_new_rays`` are used.
        atlas: Stage b's atlas, see :class:`TextureAtlas`.
        texture_map: ``(C, H_t, W_t, 3)`` the edited texture, in ``[0, 1]``.
        mask_texture_map: ``(C, H_t, W_t, 1)`` the edit mask, from
            :func:`uv.texture.create_edit_mask_from_texture_maps`.
        deformed_points: ``(M, 3)`` this frame's cloud, in the renderer's **scaled** frame.
        canonical_points: ``(M, 3)`` the checkpoint's cloud, same rows, also scaled.
        c2w: ``(4, 4)`` or ``(1, 4, 4)`` camera pose, in the scaled frame like ``dataset.c2w``.
        max_height, max_width: Tile size (``test.max_height`` / ``test.max_width``).
        bbox_hull: Optional hull from :func:`uv.bbox.load_bounding_box`, restricting
            the texture lookup to where the atlas is valid.
        bkg_color: ``(1, 1, 1, 3)`` colour composited behind the render; white when omitted.
        fuse_type: Surface fusion, see :data:`models.surface.FUSE_TYPES`.
        transfer_method: ``"deformation_aware"`` (the paper's) or ``"naive"`` (the ablation).
        project_before_transfer: Re-project the fused point onto its ray inside the transfer.
            **False**, matching the archived render: ``position-sphere-projection`` has already
            done exactly that, so this second projection is idempotent in exact arithmetic and
            only perturbs the last bits. Exposed because the transfer's own projection is step 1
            of the paper's method and an ablation that changes ``fuse_type`` needs it back.
        step: Training step handed to ``evaluate``; drives the spherical-harmonic band schedule.

    Returns:
        An :class:`EditedFrame`.
    """
    check_edit_preconditions(
        model.args,
        checkpoint_points=canonical_points.shape[0],
        deformed_points=deformed_points.shape[0],
    )
    if texture_map.shape[0] != atlas.num_charts:
        raise ValueError(
            f"texture map has {texture_map.shape[0]} charts, the atlas has {atlas.num_charts}"
        )

    device = deformed_points.device
    height, width = int(dataset.H), int(dataset.W)
    coord_scale = float(model.coord_scale)

    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    c2w = c2w.to(device)
    rays_o, rays_d, _ = dataset.get_new_rays(c2w)
    rays_o, rays_d = rays_o.to(device), rays_d.to(device)
    pix_coords = dataset.pix_coords.unsqueeze(0).to(device)

    if bkg_color is None:
        bkg_color = torch.ones(1, 1, 1, 3, device=device)

    fused: Tensor | None = None
    attn_map: Tensor | None = None
    index_map: Tensor | None = None
    surface_map = torch.zeros(1, height, width, 3, device=device)

    for top, bottom in _tile_bounds(height, max_height):
        for left, right in _tile_bounds(width, max_width):
            tile_rays_d = rays_d[:, top:bottom, left:right]
            out = model.evaluate(
                rays_o,
                tile_rays_d,
                c2w,
                pix_coords[:, top:bottom, left:right],
                step=step,
                deformed_points=deformed_points,
            )
            if fused is None:
                slots = out.attn.shape[-2]
                candidates = out.select_k_ind.shape[-1]
                channels = out.fused_features.shape[-1]
                fused = torch.zeros(1, height, width, 1, channels, device=device)
                attn_map = torch.zeros(1, height, width, slots, 1, device=device)
                index_map = torch.zeros(1, height, width, candidates, dtype=torch.long,
                                        device=device)
            fused[:, top:bottom, left:right] = out.fused_features
            attn_map[:, top:bottom, left:right] = out.attn
            index_map[:, top:bottom, left:right] = out.select_k_ind
            surface_map[:, top:bottom, left:right] = get_surface_points(
                out.cloud.points,
                out.select_k_ind,
                rays_o,
                tile_rays_d,
                out.attn,
                fuse_type,
                sphere_intersection=out.sphere_intersection,
                coord_scale=coord_scale,
                divide_by_coord_scale=bool(model.args.texture.divide_by_coord_scale),
            )

    if fused is None:  # pragma: no cover - a zero-sized frame cannot reach here
        raise ValueError("frame has no tiles; check test.max_height / test.max_width")

    rgb, _ = model.decode(fused.squeeze(-2), 1, height, width)
    rgb = _composite_background(model, rgb, attn_map, bkg_color)
    papr_rgb = torch.clamp(model.last_act(rgb), 0, 1).squeeze(0).float()

    texture_rgb = torch.zeros(height, width, 3, device=device)
    mask_image = torch.zeros(height, width, 1, device=device)

    if bbox_hull is not None:
        foreground = points_inside_bounding_box(surface_map[0], bbox_hull)
    else:
        foreground = (1 - attn_map[..., -1:, :].squeeze(-1)).squeeze() > 0.5

    fg_surface = surface_map[0][foreground]
    if fg_surface.shape[0] > 0:
        fg_attn = attn_map[0][foreground]
        fg_indices = index_map[0][foreground]
        fg_rays_o = rays_o.reshape(1, 1, 1, 3).expand(1, height, width, 3)[0][foreground]
        fg_rays_d = rays_d[0][foreground]

        canonical_surface = transfer(
            transfer_method,
            deformed_surface_points=fg_surface,
            canonical_points=canonical_points / coord_scale,
            deformed_points=deformed_points / coord_scale,
            selected_indices=fg_indices,
            attn=fg_attn,
            rays_o=fg_rays_o / coord_scale,
            rays_d=fg_rays_d,
            project=project_before_transfer,
        )

        texture_rgb[foreground] = atlas.sample_texture(canonical_surface, texture_map).clamp(0, 1)
        sampled_mask = atlas.sample_texture(canonical_surface, mask_texture_map)
        mask_image[foreground] = sampled_mask[..., :1].clamp(0, 1)

    final_rgb = (mask_image * texture_rgb + (1 - mask_image) * papr_rgb).clamp(0, 1)

    return EditedFrame(
        final_rgb=final_rgb,
        papr_rgb=papr_rgb,
        texture_rgb=texture_rgb,
        mask=mask_image,
        surface_points=surface_map,
        foreground=foreground,
    )


def to_uint8(image: Tensor) -> np.ndarray:
    """``[0, 1]`` float tensor to an 8-bit array, truncating rather than rounding."""
    return (image.detach().cpu().numpy() * 255).astype(np.uint8)

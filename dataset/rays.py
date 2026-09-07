from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

#: Guards the perspective divide. Points closer than this to the image plane project nowhere
#: meaningful; the value only keeps the result finite so callers can mask on the returned depth.
_MIN_DEPTH = 1e-6


def _check_image_size(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError(f"Image size must be positive, got height={height}, width={width}")


def _check_focal(focal_x: float, focal_y: float) -> None:
    if focal_x <= 0 or focal_y <= 0:
        raise ValueError(f"Focal lengths must be positive, got fx={focal_x}, fy={focal_y}")


def _resolve_principal_point(
    height: int, width: int, principal_point: tuple[float, float] | None
) -> tuple[float, float]:
    if principal_point is None:
        return width / 2.0, height / 2.0
    cx, cy = principal_point
    return float(cx), float(cy)


def build_intrinsics(
    focal_x: float,
    focal_y: float,
    height: int,
    width: int,
    *,
    principal_point: tuple[float, float] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build the pinhole intrinsics matrix ``K`` for an image of size ``(height, width)``.

    The principal point defaults to the image centre ``(W/2, H/2)``. Every shipped loader produces
    centred intrinsics; an off-centre principal point stays self-consistent within this module, but
    a renderer that hardcodes the centre will disagree with it.

    Returns:
        ``(3, 3)`` tensor ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
    """
    _check_image_size(height, width)
    _check_focal(focal_x, focal_y)
    cx, cy = _resolve_principal_point(height, width, principal_point)
    k = torch.zeros(3, 3, device=device, dtype=dtype)
    k[0, 0] = float(focal_x)
    k[1, 1] = float(focal_y)
    k[0, 2] = cx
    k[1, 2] = cy
    k[2, 2] = 1.0
    return k


def pixel_centers(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Half-integer ``(u, v)`` image coordinates of every pixel centre.

    Returns:
        ``(height, width, 2)`` tensor; ``out[j, i] == (i + 0.5, j + 0.5)``.
    """
    _check_image_size(height, width)
    u = torch.arange(width, device=device, dtype=dtype) + 0.5
    v = torch.arange(height, device=device, dtype=dtype) + 0.5
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([grid_u, grid_v], dim=-1)


def camera_ray_directions(
    height: int,
    width: int,
    focal_x: float,
    focal_y: float,
    *,
    principal_point: tuple[float, float] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Per-pixel ray directions in the OpenGL camera frame, on the ``z = -1`` plane.

    The z component is exactly ``-1`` rather than unit-normalised, so scaling one of these vectors
    by ``t`` moves ``t`` along the optical axis: the parameter is z-depth, not path length.

    Returns:
        ``(height, width, 3)`` tensor.
    """
    _check_image_size(height, width)
    _check_focal(focal_x, focal_y)
    cx, cy = _resolve_principal_point(height, width, principal_point)
    uv = pixel_centers(height, width, device=device, dtype=dtype)
    x = (uv[..., 0] - cx) / float(focal_x)
    y = -(uv[..., 1] - cy) / float(focal_y)
    return torch.stack([x, y, -torch.ones_like(x)], dim=-1)


def _affine(coords: Tensor, matrix: Tensor, homogeneous: float) -> Tensor:
    if coords.shape[-1] != 3:
        raise ValueError(f"Expected coordinates with trailing dim 3, got {tuple(coords.shape)}")
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Expected a (..., 4, 4) matrix, got {tuple(matrix.shape)}")
    pad = torch.full_like(coords[..., :1], homogeneous)
    hom = torch.cat([coords, pad], dim=-1)
    return torch.einsum("...ij,...j->...i", matrix, hom)[..., :3]


def transform_points(points: Tensor, matrix: Tensor) -> Tensor:
    """Apply a 4x4 transform to points ``(..., 3)``; the translation column acts.

    A batched ``matrix`` must already carry singleton dims that broadcast against the leading dims
    of ``points`` -- e.g. ``c2w[:, None, None]`` for an ``(N, H, W, 3)`` grid. Requiring the caller
    to spell that out keeps a mis-aligned batch a shape error rather than a silent broadcast.
    """
    return _affine(points, matrix, 1.0)


def transform_directions(directions: Tensor, matrix: Tensor) -> Tensor:
    """Apply the linear part of a 4x4 transform to directions ``(..., 3)``.

    The translation column is suppressed. Broadcasting rules match :func:`transform_points`.
    """
    return _affine(directions, matrix, 0.0)


def invert_pose(c2w: Tensor) -> Tensor:
    """Invert a camera-to-world matrix, giving world-to-camera.

    A general inverse rather than the rigid ``[R^T | -R^T t]`` shortcut, because ``coord_scale``
    poses carry a uniform scale in the rotation block and are not rigid.
    """
    if c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected a (..., 4, 4) pose, got {tuple(c2w.shape)}")
    return torch.linalg.inv(c2w)


def scale_pose(c2w: Tensor, scale: float) -> Tensor:
    """Left-multiply by ``diag(scale, scale, scale, 1)``, scaling the whole world.

    This scales the rotation block too, which is what makes ``raw_directions`` come out ``scale``
    times longer. See the module docstring.
    """
    if scale <= 0:
        raise ValueError(f"Pose scale must be positive, got {scale}")
    if c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected a (..., 4, 4) pose, got {tuple(c2w.shape)}")
    scaled = c2w.clone()
    scaled[..., :3, :] = scaled[..., :3, :] * scale
    return scaled


def opencv_to_opengl_pose(c2w: Tensor) -> Tensor:
    """Negate the y and z camera axes of a camera-to-world matrix.

    Equivalent to right-multiplying by ``diag(1, -1, -1, 1)``: it rewrites the camera basis
    (y-down, +z-forward) as (y-up, -z-forward) while leaving the camera centre alone. The
    transform is its own inverse, hence :func:`opengl_to_opencv_pose` below.
    """
    if c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected a (..., 4, 4) pose, got {tuple(c2w.shape)}")
    flipped = c2w.clone()
    flipped[..., :, 1:3] = -flipped[..., :, 1:3]
    return flipped


def opengl_to_opencv_pose(c2w: Tensor) -> Tensor:
    """Inverse of :func:`opencv_to_opengl_pose`, which is the same involution."""
    return opencv_to_opengl_pose(c2w)


@dataclass(frozen=True)
class RayBundle:
    """Rays for a batch of ``N`` cameras over an ``H x W`` pixel grid.

    Attributes:
        origins: ``(N, 3)`` camera centres in world space; one per camera, shared by its pixels.
        directions: ``(N, H, W, 3)`` unit-norm world-space directions.
        raw_directions: ``(N, H, W, 3)`` world-space directions whose length is set so that
            ``origins + t * raw_directions`` advances ``t`` along the optical axis. z-depth
            parameterises these; path length parameterises :attr:`directions`.
        camera_plane: ``(N, H, W, 3)`` the same rays in the *camera* frame, on ``z = -1``. Pose
            independent, so it is a broadcast view of a single grid rather than ``N`` copies.
    """

    origins: Tensor
    directions: Tensor
    raw_directions: Tensor
    camera_plane: Tensor

    def __post_init__(self) -> None:
        if self.origins.ndim != 2 or self.origins.shape[-1] != 3:
            raise ValueError(f"origins must be (N, 3), got {tuple(self.origins.shape)}")
        n = self.origins.shape[0]
        for name in ("directions", "raw_directions", "camera_plane"):
            grid: Tensor = getattr(self, name)
            if grid.ndim != 4 or grid.shape[0] != n or grid.shape[-1] != 3:
                raise ValueError(f"{name} must be ({n}, H, W, 3), got {tuple(grid.shape)}")
        if self.directions.shape != self.raw_directions.shape:
            raise ValueError("directions and raw_directions must have the same shape")
        if self.directions.shape != self.camera_plane.shape:
            raise ValueError("directions and camera_plane must have the same shape")

    @property
    def num_cameras(self) -> int:
        return self.origins.shape[0]

    @property
    def height(self) -> int:
        return self.directions.shape[1]

    @property
    def width(self) -> int:
        return self.directions.shape[2]


def _as_batched_poses(c2w: Tensor) -> Tensor:
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)
    if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected poses of shape (4, 4) or (N, 4, 4), got {tuple(c2w.shape)}")
    return c2w


def generate_rays(
    height: int,
    width: int,
    focal_x: float,
    focal_y: float,
    c2w: Tensor,
    *,
    principal_point: tuple[float, float] | None = None,
) -> RayBundle:
    """Generate world-space rays for every pixel of every supplied pose.

    ``c2w`` may be ``(4, 4)`` or ``(N, 4, 4)``; the result is always batched, with ``N = 1`` for a
    single pose, so downstream indexing does not depend on how the caller spelled the input.
    Device and dtype follow ``c2w``.
    """
    poses = _as_batched_poses(c2w)
    dirs_cam = camera_ray_directions(
        height,
        width,
        focal_x,
        focal_y,
        principal_point=principal_point,
        device=poses.device,
        dtype=poses.dtype,
    )
    raw = transform_directions(dirs_cam, poses[:, None, None])
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    unit = raw / norm
    plane = dirs_cam.unsqueeze(0).expand(poses.shape[0], -1, -1, -1)
    return RayBundle(
        origins=poses[:, :3, 3].clone(),
        directions=unit,
        raw_directions=raw,
        camera_plane=plane,
    )


def project_points(points_world: Tensor, c2w: Tensor, intrinsics: Tensor) -> tuple[Tensor, Tensor]:
    """Project world points to image coordinates -- the inverse of :func:`generate_rays`.

    The OpenGL-to-OpenCV axis flip that ``K`` expects happens here, so callers never apply ``K`` to
    OpenGL camera coordinates themselves.

    Batched ``c2w`` and ``intrinsics`` follow the broadcasting rule of :func:`transform_points`.

    Returns:
        ``(uv, depth)`` where ``uv`` is ``(..., 2)`` in the half-integer pixel convention and
        ``depth`` is ``(...)``, positive in front of the camera. ``uv`` is meaningless wherever
        ``depth`` is non-positive, so mask on it rather than trusting the coordinates.
    """
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3) intrinsics, got {tuple(intrinsics.shape)}")
    cam = transform_points(points_world, invert_pose(c2w))
    depth = -cam[..., 2]
    safe = torch.where(depth.abs() < _MIN_DEPTH, torch.full_like(depth, _MIN_DEPTH), depth)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    u = fx * cam[..., 0] / safe + cx
    v = -fy * cam[..., 1] / safe + cy
    return torch.stack([u, v], dim=-1), depth


def slice_patch(grid: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """Crop an axis-aligned window from a ``(..., H, W, C)`` grid.

    The window must lie wholly inside the image; a partly out-of-range crop is an error rather than
    a silently short slice, which is how a mis-sized patch would otherwise reach the loss.
    """
    if grid.ndim < 3:
        raise ValueError(f"Expected a (..., H, W, C) grid, got {tuple(grid.shape)}")
    _check_image_size(height, width)
    full_h, full_w = grid.shape[-3], grid.shape[-2]
    if top < 0 or left < 0 or top + height > full_h or left + width > full_w:
        raise ValueError(
            f"Patch [{top}:{top + height}, {left}:{left + width}] does not fit in "
            f"a {full_h}x{full_w} image"
        )
    return grid[..., top : top + height, left : left + width, :]


def extract_patch(rays: RayBundle, top: int, left: int, height: int, width: int) -> RayBundle:
    """Crop every ray grid in a bundle to the same window. Origins are per-camera, so they carry
    over unchanged.

    The result is a view: its rays are the corresponding slice of the full-image bundle, equal
    element for element rather than regenerated at patch resolution.
    """
    return RayBundle(
        origins=rays.origins,
        directions=slice_patch(rays.directions, top, left, height, width),
        raw_directions=slice_patch(rays.raw_directions, top, left, height, width),
        camera_plane=slice_patch(rays.camera_plane, top, left, height, width),
    )


def random_patch_origin(
    image_size: tuple[int, int],
    patch_size: tuple[int, int],
    *,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Sample a uniform top-left corner for a patch of ``patch_size`` inside ``image_size``.

    Both bounds are inclusive: an origin of ``H - patch_h`` is a legal crop and is sampled with the
    same probability as any other, so the last row and column of an image are as likely to appear
    in a patch as the interior.
    """
    full_h, full_w = image_size
    patch_h, patch_w = patch_size
    _check_image_size(full_h, full_w)
    _check_image_size(patch_h, patch_w)
    if patch_h > full_h or patch_w > full_w:
        raise ValueError(f"Patch {patch_h}x{patch_w} does not fit in image {full_h}x{full_w}")
    top = torch.randint(0, full_h - patch_h + 1, (1,), generator=generator).item()
    left = torch.randint(0, full_w - patch_w + 1, (1,), generator=generator).item()
    return int(top), int(left)

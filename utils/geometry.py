from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

#: Additive epsilon of the attention-feature normalisation. Part of the trained numerics.
FEATURE_NORM_EPS = 1e-6

#: Lower clamp on a norm before dividing by it. Small enough to be inert for real geometry.
UNIT_EPS = 1e-8

#: Guards a division by a camera-space depth of exactly zero in the pinhole projection.
DEPTH_EPS = 1e-6

__all__ = [
    "DEPTH_EPS",
    "FEATURE_NORM_EPS",
    "UNIT_EPS",
    "RayProjection",
    "RectifiedPoints",
    "broadcast_ray_origins",
    "camera_to_world",
    "construct_coord_frame",
    "invert_rigid_transform",
    "normalize_vector",
    "perspective_projection",
    "project_point_onto_ray",
    "project_points_to_image",
    "ray_sphere_intersection",
    "ray_sphere_intersection_t",
    "rectify_points",
    "transform_points",
    "world_to_camera",
]


def _require_vec3(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim == 0 or tensor.shape[-1] != 3:
        raise ValueError(f"{name} must have a trailing dimension of 3, got {tuple(tensor.shape)}")


def _unit(vectors: Tensor, eps: float = UNIT_EPS) -> Tensor:
    """Normalise by a floor-clamped norm, so unit input survives bit-for-bit."""
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(eps)


def normalize_vector(vectors: Tensor, eps: float = FEATURE_NORM_EPS) -> Tensor:
    """Scale to (approximately) unit length by dividing by ``norm + eps``.

    The additive epsilon shrinks the result by a relative ``eps`` even for exactly unit input. That
    bias is baked into the trained attention features, so this is the normalisation to use wherever
    the result feeds a network; pure geometry uses the floor-clamped form instead.
    """
    _require_vec3(vectors, "vectors")
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + eps)


def broadcast_ray_origins(ray_origins: Tensor, ray_directions: Tensor) -> Tensor:
    """Expand a per-view or per-scene origin to one origin per direction.

    Pinhole views share a single origin across the whole image, so origins arrive as ``(3,)`` or
    ``(N, 3)`` while directions are ``(N, H, W, 3)``. Trailing-dimension broadcasting would silently
    align ``N`` with ``W``, so the leading-batch convention is resolved explicitly here and any
    shape it cannot account for is an error rather than a guess.
    """
    _require_vec3(ray_origins, "ray_origins")
    _require_vec3(ray_directions, "ray_directions")

    if ray_origins.shape == ray_directions.shape:
        return ray_origins
    if ray_origins.ndim == 1:
        return ray_origins.expand(ray_directions.shape)
    if ray_origins.ndim == 2 and ray_origins.shape[0] == ray_directions.shape[0]:
        lead = (ray_origins.shape[0],) + (1,) * (ray_directions.ndim - 2) + (3,)
        return ray_origins.reshape(lead).expand(ray_directions.shape)
    if (
        ray_origins.ndim == ray_directions.ndim - 1
        and ray_origins.shape == ray_directions.shape[1:]
    ):
        return ray_origins.unsqueeze(0).expand(ray_directions.shape)
    raise ValueError(
        f"ray_origins {tuple(ray_origins.shape)} cannot be matched to ray_directions "
        f"{tuple(ray_directions.shape)}"
    )


def _as_center(center: Tensor | Sequence[float] | None, like: Tensor) -> Tensor:
    if center is None:
        return torch.zeros(3, dtype=like.dtype, device=like.device)
    if not isinstance(center, Tensor):
        center = torch.as_tensor(center, dtype=like.dtype, device=like.device)
    center = center.to(dtype=like.dtype, device=like.device)
    if center.shape != (3,):
        raise ValueError(f"sphere center must have shape (3,), got {tuple(center.shape)}")
    return center


def _solve_ray_sphere(
    ray_origins: Tensor,
    ray_directions: Tensor,
    center: Tensor | Sequence[float] | None,
    radius: float | Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Shared core: returns ``(t, hit, origins, unit_directions)`` with everything broadcast."""
    _require_vec3(ray_origins, "ray_origins")
    _require_vec3(ray_directions, "ray_directions")

    origins = broadcast_ray_origins(ray_origins, ray_directions)
    directions = F.normalize(ray_directions, dim=-1, eps=UNIT_EPS)

    sphere_center = _as_center(center, ray_directions)
    radius_t = torch.as_tensor(radius, dtype=ray_directions.dtype, device=ray_directions.device)

    offset = origins - sphere_center
    along = torch.sum(offset * directions, dim=-1)
    radial_sq = torch.sum(offset * offset, dim=-1)

    discriminant = along * along - radial_sq + radius_t * radius_t
    t = -along + torch.sqrt(torch.clamp(discriminant, min=0.0))
    return torch.clamp(t, min=0.0), discriminant >= 0, origins, directions


def ray_sphere_intersection_t(
    ray_origins: Tensor,
    ray_directions: Tensor,
    center: Tensor | Sequence[float] | None = None,
    radius: float | Tensor = 1.0,
) -> tuple[Tensor, Tensor]:
    """Solve for the *far* root of the ray/sphere quadratic.

    Returns ``(t, hit)`` where ``t`` is ``(...)`` -- the arc length along the **unit-normalised**
    direction -- and ``hit`` is the boolean mask of rays that actually meet the sphere.

    The far root is the right one for a background sphere: the camera sits inside it, so the
    forward exit point is what a ray that escapes the point cloud should land on.

    Two clamps make the result total, and both are load-bearing for the editing path, which fuses
    this point into every pixel including ones whose geometry is degenerate:

    * a negative discriminant is clamped to zero, so a missing ray returns its point of closest
      approach to the sphere centre instead of a NaN;
    * ``t`` is clamped to be non-negative, so a ray whose exit point lies behind it returns its own
      origin rather than a point mirrored through the camera.

    Use ``hit`` to tell a real intersection from either fallback.
    """
    t, hit, _, _ = _solve_ray_sphere(ray_origins, ray_directions, center, radius)
    return t, hit


def ray_sphere_intersection(
    ray_origins: Tensor,
    ray_directions: Tensor,
    center: Tensor | Sequence[float] | None = None,
    radius: float | Tensor = 1.0,
) -> Tensor:
    """Point where each ray leaves the sphere, shaped like ``ray_directions``.

    See :func:`ray_sphere_intersection_t` for the root choice and the two clamps.
    """
    t, _, origins, directions = _solve_ray_sphere(ray_origins, ray_directions, center, radius)
    return origins + t.unsqueeze(-1) * directions


@dataclass(frozen=True)
class RayProjection:
    """Decomposition of ``point - origin`` into its along-ray and across-ray parts."""

    #: Signed arc length from the ray origin to the foot, ``(..., 1)``. Negative behind the camera.
    t: Tensor
    #: Foot of the perpendicular, ``origin + t * direction``, ``(..., 3)``.
    foot: Tensor
    #: Along-ray component ``t * direction``, ``(..., 3)``.
    parallel: Tensor
    #: Across-ray component ``point - foot``, ``(..., 3)``.
    perpendicular: Tensor
    #: ``|parallel|``, ``(..., 1)``. Unsigned -- see :func:`project_point_onto_ray`.
    projected_distance: Tensor
    #: ``|perpendicular|``, ``(..., 1)``. The point's distance to the infinite ray.
    distance_to_ray: Tensor


def project_point_onto_ray(
    points: Tensor,
    ray_origins: Tensor,
    ray_directions: Tensor,
    normalize: bool = True,
) -> RayProjection:
    """Orthogonally project points onto rays -- the paper's on-ray operator.

    For a ray ``r(t) = o + t*d`` with ``|d| = 1`` and a point ``p``, this returns the unique ``t``
    minimising ``|r(t) - p|``, namely ``t = (p - o) . d``, together with the foot ``r(t)`` and the
    orthogonal split ``p - o = parallel + perpendicular``. ``perpendicular`` is exactly the residual
    that the on-ray loss drives to zero, and ``distance_to_ray`` is the radial coordinate the
    proximity attention scores points by, while ``t`` is its depth coordinate.

    It is also step 1 of the deformation-aware canonical correspondence: a surface point fused from
    attention weights is first pulled back onto its own ray -- which removes the drift the weighted
    average introduces off-ray -- before it is matched against the canonical point cloud. Skipping
    the projection makes the correspondence depend on how the attention mass happened to spread
    laterally, not on where the surface is.

    Shapes broadcast in the ordinary trailing-dimension sense against a common ``(..., 3)``: for
    ``K`` candidate points per pixel, pass ``points`` as ``(N, H, W, K, 3)``, ``ray_directions`` as
    ``(N, H, W, 1, 3)`` and ``ray_origins`` as ``(N, 1, 1, 1, 3)``. Reshaping is left to the caller
    because only the caller knows which axis is the view axis.

    ``projected_distance`` is ``|t|``, so it cannot distinguish a point in front of the camera from
    its mirror image behind it. Anything that needs the sign must read ``t``.

    Args:
        points: ``(..., 3)`` points to project.
        ray_origins: ``(..., 3)`` ray origins, broadcastable against ``points``.
        ray_directions: ``(..., 3)`` ray directions, broadcastable against ``points``.
        normalize: Rescale ``ray_directions`` to unit norm first. Leave it on unless the caller has
            already normalised -- the returned ``t`` is only a distance for a unit direction.
    """
    _require_vec3(points, "points")
    _require_vec3(ray_origins, "ray_origins")
    _require_vec3(ray_directions, "ray_directions")

    directions = _unit(ray_directions) if normalize else ray_directions
    offset = points - ray_origins
    t = torch.sum(offset * directions, dim=-1, keepdim=True)
    parallel = t * directions
    perpendicular = offset - parallel
    return RayProjection(
        t=t,
        foot=ray_origins + parallel,
        parallel=parallel,
        perpendicular=perpendicular,
        projected_distance=torch.norm(parallel, dim=-1, keepdim=True),
        distance_to_ray=torch.norm(perpendicular, dim=-1, keepdim=True),
    )


def construct_coord_frame(z: Tensor, y: Tensor) -> Tensor:
    """Build right-handed orthonormal frames from a z axis and a y hint.

    ``z`` fixes the third axis. ``y`` is only a hint: its component along ``z`` is removed before it
    becomes the second axis, so any ``y`` not parallel to ``z`` yields the same frame up to that
    projection. The first axis is ``y x z``, which makes the frame right-handed (``det = +1``).

    Args:
        z: ``(..., 3)`` third axis, need not be normalised.
        y: ``(..., 3)`` up hint, need not be normalised or orthogonal to ``z``. Broadcasts with
            ``z``.

    Returns:
        ``(..., 3, 3)`` with the axes as **columns**, i.e. the rotation taking a vector expressed in
        the new frame to the frame ``z`` and ``y`` were given in. Transpose it for the inverse.

    A ``y`` parallel to ``z`` leaves the frame underdetermined; the norms are floor-clamped so the
    result stays finite rather than NaN, but it is arbitrary. Callers that can hit this -- deformed
    point clouds trained without the on-ray loss produce near-zero ray directions -- should treat
    those entries as invalid rather than rely on the fallback.
    """
    _require_vec3(z, "z")
    _require_vec3(y, "y")
    z, y = torch.broadcast_tensors(z, y)

    axis_z = _unit(z)
    axis_x = _unit(torch.linalg.cross(y, z, dim=-1))
    up = y - torch.sum(y * axis_z, dim=-1, keepdim=True) * axis_z
    axis_y = _unit(up)
    return torch.stack((axis_x, axis_y, axis_z), dim=-1)


@dataclass(frozen=True)
class RectifiedPoints:
    """Points expressed in the per-ray canonical frame, plus the transform that produced them."""

    #: ``(*, m, n, 3)`` points in the rectified frame.
    points: Tensor
    #: ``(*, m, 3, 3)`` rotation taking world vectors to the rectified frame.
    rotation: Tensor
    #: ``(*, m, 3, 1)`` translation completing that transform.
    translation: Tensor
    #: ``(*, m)`` arc length the frame origin was slid along the ray. Zero unless ``translate``.
    shift: Tensor


def rectify_points(
    points: Tensor,
    ray_origins: Tensor,
    ray_directions: Tensor,
    translate: bool = False,
    ts: Tensor | None = None,
    t_min: float = 0.0,
    t_max: float = 1e6,
) -> RectifiedPoints:
    """Rewrite each ray's candidate points in that ray's own frame.

    The rigid transform sends the ray direction to ``+z`` and the ray origin to the origin, so a
    point's rectified coordinates read directly as (across-ray offset, along-ray depth). Encoding
    points this way makes the attention keys invariant to camera pose, which is what the editing
    feature types depend on: a point edited in world space keeps the same key relative to whichever
    ray sees it.

    With ``translate``, the frame origin is additionally slid forward to the nearest admissible
    projection, so the closest point of each ray sits at ``z = 0``. That removes the scene's
    absolute depth from the encoding -- otherwise every key carries the distance from the camera to
    the object, which does not generalise across scenes or across an edit that moves geometry
    toward or away from the camera.

    Args:
        points: ``(*, m, n, 3)`` -- ``n`` candidate points for each of ``m`` rays.
        ray_origins: ``(*, m, 3)``.
        ray_directions: ``(*, m, 3)``, unit norm.
        translate: Slide the frame origin as described above.
        ts: ``(*, m, n)`` along-ray coordinate of each point, required when ``translate`` is set.
            Values outside ``[t_min, t_max]`` are ignored when picking the nearest one; a ray with
            no admissible point is not slid at all.
        t_min: Lower admissible bound on ``ts``.
        t_max: Upper admissible bound on ``ts``. The default is the sentinel depth the renderer
            parks background points at, so they never drag the frame origin out of the scene.

    Returns:
        A :class:`RectifiedPoints`. Reuse ``rotation``/``translation`` to carry other per-point
        vectors (normals, attention offsets) into the same frame.
    """
    _require_vec3(points, "points")
    _require_vec3(ray_origins, "ray_origins")
    _require_vec3(ray_directions, "ray_directions")
    if points.ndim != ray_directions.ndim + 1:
        raise ValueError(
            f"points {tuple(points.shape)} must have exactly one more dimension than "
            f"ray_directions {tuple(ray_directions.shape)}"
        )

    up_hint = torch.zeros_like(ray_directions)
    up_hint[..., 1] = 1.0
    frame_n2w = construct_coord_frame(z=ray_directions, y=up_hint)

    if translate:
        if ts is None:
            raise ValueError("translate=True requires ts, the along-ray coordinate of each point")
        if ts.shape != points.shape[:-1]:
            raise ValueError(
                f"ts {tuple(ts.shape)} must match points {tuple(points.shape[:-1])} without the "
                "coordinate axis"
            )
        admissible = (ts >= t_min) & (ts <= t_max)
        infinity = torch.full_like(ts, float("inf"))
        shift = torch.where(admissible, ts, infinity).min(dim=-1, keepdim=True).values
        shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
        origins = ray_origins + shift * ray_directions
    else:
        shift = torch.zeros(
            (*ray_origins.shape[:-1], 1), dtype=ray_origins.dtype, device=ray_origins.device
        )
        origins = ray_origins

    rotation = frame_n2w.transpose(-1, -2)
    translation = -(rotation @ origins.unsqueeze(-1))
    rectified = rotation.unsqueeze(-3) @ points.unsqueeze(-1) + translation.unsqueeze(-3)
    return RectifiedPoints(
        points=rectified.squeeze(-1),
        rotation=rotation,
        translation=translation,
        shift=shift.squeeze(-1),
    )


def _align_matrix(matrix: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Resolve the leading-batch convention between a transform stack and a coordinate tensor."""
    if matrix.ndim < 2 or matrix.shape[-2:] != (4, 4):
        raise ValueError(f"matrix must be (4, 4) or (B, 4, 4), got {tuple(matrix.shape)}")
    if matrix.ndim == 2:
        return matrix.reshape((1,) * (coords.ndim - 1) + (4, 4)), coords
    if matrix.ndim > 3:
        raise ValueError(f"matrix must be (4, 4) or (B, 4, 4), got {tuple(matrix.shape)}")

    batch = matrix.shape[0]
    if coords.ndim == 2:
        return matrix.reshape(batch, 1, 4, 4), coords.unsqueeze(0)
    if coords.shape[0] not in (1, batch):
        raise ValueError(
            f"coords {tuple(coords.shape)} has no leading axis matching the {batch} transforms"
        )
    return matrix.reshape((batch,) + (1,) * (coords.ndim - 2) + (4, 4)), coords


def transform_points(coords: Tensor, matrix: Tensor, vector: bool = False) -> Tensor:
    """Apply 4x4 homogeneous transforms to 3D coordinates.

    ``vector=True`` transforms directions (homogeneous ``w = 0``, translation ignored);
    ``vector=False`` transforms positions.

    Batching follows the shape of ``matrix``:

    * ``(4, 4)`` -- applied to every coordinate; output shape equals ``coords``.
    * ``(B, 4, 4)`` with ``coords`` of rank >= 3 -- transform ``b`` applies to ``coords[b]``, so
      ``(B, H, W, 3)`` stays ``(B, H, W, 3)``. A leading axis of 1 broadcasts.
    * ``(B, 4, 4)`` with ``coords`` of rank 2 -- a flat ``(K, 3)`` point set has no view axis, so
      every transform is applied to every point and the result is ``(B, K, 3)``.
    """
    _require_vec3(coords, "coords")
    if coords.ndim < 2:
        raise ValueError(f"coords must have a batch dimension, got {tuple(coords.shape)}")

    pad = torch.zeros_like(coords[..., :1]) if vector else torch.ones_like(coords[..., :1])
    homogeneous = torch.cat([coords, pad], dim=-1)
    aligned, homogeneous = _align_matrix(matrix, homogeneous)
    return (homogeneous.unsqueeze(-2) * aligned).sum(dim=-1)[..., :3]


def camera_to_world(coords: Tensor, c2w: Tensor, vector: bool = True) -> Tensor:
    """Map camera-space coordinates to world space. See :func:`transform_points` for batching."""
    return transform_points(coords, c2w, vector=vector)


def world_to_camera(coords: Tensor, c2w: Tensor, vector: bool = True) -> Tensor:
    """Map world-space coordinates to camera space, inverting the camera-to-world transform.

    The inverse is taken generally rather than assuming rigidity, because some archived pose
    conventions carry a scene scale in the rotation block. Use :func:`invert_rigid_transform` when
    the transform is known to be rigid and the extra accuracy or speed matters.
    """
    return transform_points(coords, torch.linalg.inv(c2w), vector=vector)


def invert_rigid_transform(matrix: Tensor, tolerance: float = 1e-4) -> Tensor:
    """Invert a rigid ``[R | t]`` transform analytically as ``[R^T | -R^T t]``.

    Raises if the rotation block is not orthonormal or the bottom row is not ``[0, 0, 0, 1]``,
    rather than returning a wrong answer for a matrix that carries scale or shear.
    """
    if matrix.ndim < 2 or matrix.shape[-2:] != (4, 4):
        raise ValueError(f"matrix must be (..., 4, 4), got {tuple(matrix.shape)}")

    rotation = matrix[..., :3, :3]
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    if not torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=tolerance):
        raise ValueError("matrix is not rigid: its 3x3 block is not orthonormal")

    bottom = matrix[..., 3, :]
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=matrix.dtype, device=matrix.device)
    if not torch.allclose(bottom, expected.expand_as(bottom), atol=tolerance):
        raise ValueError("matrix is not affine: its bottom row is not [0, 0, 0, 1]")

    inverse = torch.zeros_like(matrix)
    inverse[..., :3, :3] = rotation.transpose(-1, -2)
    inverse[..., :3, 3] = -(rotation.transpose(-1, -2) @ matrix[..., :3, 3:]).squeeze(-1)
    inverse[..., 3, 3] = 1.0
    return inverse


def perspective_projection(
    coords_cam: Tensor,
    fx: float | Tensor,
    fy: float | Tensor,
    cx: float | Tensor,
    cy: float | Tensor,
    width: int | float,
) -> Tensor:
    """Pinhole-project camera-space points to pixel coordinates.

    Args:
        coords_cam: ``(..., 3)`` points already in camera space.
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.
        width: Image width, used by the horizontal flip below.

    Returns:
        ``(..., 2)`` pixel coordinates ``(u, v)``, with ``u`` to the right and ``v`` downward.

    Two sign conventions are folded into the arithmetic and must stay: the camera looks down its
    own ``-z``, and the image ``u`` axis runs opposite to the camera ``x`` axis, so ``u`` is
    mirrored through ``width``. Points at or behind the pinhole are not culled here -- their
    ``u, v`` are meaningless and the caller is expected to mask on ``coords_cam[..., 2]``. A small
    epsilon is added to the depth purely so an exactly-zero depth yields a huge coordinate instead
    of an infinity; at scene scale it is otherwise inert.
    """
    _require_vec3(coords_cam, "coords_cam")
    inv_depth = 1.0 / (coords_cam[..., 2] + DEPTH_EPS)
    u = width - (coords_cam[..., 0] * fx * inv_depth + cx)
    v = coords_cam[..., 1] * fy * inv_depth + cy
    return torch.stack([u, v], dim=-1)


def project_points_to_image(
    points: Tensor,
    c2w: Tensor,
    fx: float | Tensor,
    fy: float | Tensor,
    cx: float | Tensor,
    cy: float | Tensor,
    width: int | float,
    vector: bool = False,
) -> tuple[Tensor, Tensor]:
    """World points to pixel coordinates, returning the camera-space points as well.

    The camera-space tensor is not a by-product to discard: its ``z`` is the depth the renderer
    sorts and culls candidate points by, and recomputing it would double the transform cost on the
    hot path.

    Returns:
        ``(uv, coords_cam)`` of shapes ``(..., 2)`` and ``(..., 3)``. With ``c2w`` of shape
        ``(B, 4, 4)`` and ``points`` of shape ``(K, 3)`` both gain a leading ``B`` -- see
        :func:`transform_points`.
    """
    coords_cam = world_to_camera(points, c2w, vector=vector)
    return perspective_projection(coords_cam, fx, fy, cx, cy, width), coords_cam

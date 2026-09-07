from __future__ import annotations

import torch

from utils.geometry import project_point_onto_ray

__all__ = [
    "project_onto_ray",
    "deformation_aware_transfer",
    "naive_transfer",
    "TRANSFER_METHODS",
]

TRANSFER_METHODS = ("deformation_aware", "naive")


def project_onto_ray(
    points: torch.Tensor,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    *,
    clamp_behind: bool = False,
) -> torch.Tensor:
    """Orthogonally project points onto their rays -- step 1 of the transfer.

    A valid ray--surface intersection lies on its ray by construction, but the attention-weighted
    average that produces it does not, so this re-imposes the constraint exactly.

    ``clamp_behind`` defaults to **False**, matching the implementation that produced the paper's
    figures. The paper's text specifies the clamped closest-point projection
    ``t* = max(0, d . (x - o))``, but the archived edit path projects inside the surface-point fuse
    (``position-sphere-projection``) with no clamp, and feeds that result straight to the transfer.
    Clamping here would disagree with every released checkpoint for any sample that lands behind the
    camera. The clamped form is available for anyone reproducing the text rather than the figures;
    the divergence is recorded in this docstring, which is the only place it
    now lives.

    Projection is idempotent, so applying it to an already-projected point is harmless.
    """
    if points.shape[-1] != 3:
        raise ValueError(f"points must have a trailing dimension of 3, got {tuple(points.shape)}")

    projection = project_point_onto_ray(points, rays_o, rays_d)
    if not clamp_behind:
        return projection.foot
    direction = projection.parallel / projection.t.clamp(min=torch.finfo(points.dtype).eps)
    return rays_o + projection.t.clamp_min(0.0) * direction


def _normalise_weights(attn: torch.Tensor, k: int) -> torch.Tensor:
    """Reduce attention to ``(..., k)`` over neighbours and renormalise.

    Attention may carry a trailing singleton channel and may include a background slot appended
    after the ``k`` point neighbours. The background slot is dropped rather than folded in: it has no
    canonical counterpart, so including it would bias the displacement toward zero by exactly the
    background's share of the mass.
    """
    if attn.shape[-1] == 1 and attn.dim() >= 2:
        attn = attn.squeeze(-1)
    if attn.shape[-1] < k:
        raise ValueError(f"attention has {attn.shape[-1]} slots, fewer than k={k}")
    attn = attn[..., :k]
    total = attn.sum(dim=-1, keepdim=True)
    return attn / (total + 1e-8)


def deformation_aware_transfer(
    deformed_surface_points: torch.Tensor,
    canonical_points: torch.Tensor,
    deformed_points: torch.Tensor,
    selected_indices: torch.Tensor,
    attn: torch.Tensor,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    *,
    project: bool = True,
) -> torch.Tensor:
    """The paper's edit-time transfer from deformed to canonical space.

    Args:
        deformed_surface_points: ``(N, 3)`` intersections predicted in deformed space.
        canonical_points: ``(M, 3)`` canonical point cloud.
        deformed_points: ``(M, 3)`` deformed point cloud, **same ordering** as ``canonical_points``.
        selected_indices: ``(N, k)`` indices of the top-k neighbours chosen in deformed space.
        attn: ``(N, k)`` or ``(N, k, 1)`` deformed-space attention weights, optionally with a
            trailing background slot.
        rays_o: ``(N, 3)`` ray origins.
        rays_d: ``(N, 3)`` ray directions.
        project: apply the on-ray projection (step 1). Exposed so the ablation can disable it;
            leaving it off is what produces the drift the paper reports.

    Returns:
        ``(N, 3)`` intersections expressed in canonical space, ready for UV lookup.
    """
    if canonical_points.shape != deformed_points.shape:
        raise ValueError(
            "canonical and deformed clouds must correspond one-to-one, got "
            f"{tuple(canonical_points.shape)} and {tuple(deformed_points.shape)}"
        )

    k = selected_indices.shape[-1]
    weights = _normalise_weights(attn, k).unsqueeze(-1)

    x_def = deformed_surface_points
    if project:
        x_def = project_onto_ray(x_def, rays_o, rays_d)

    displacement = canonical_points[selected_indices] - deformed_points[selected_indices]
    return x_def + (weights * displacement).sum(dim=-2)


def naive_transfer(
    canonical_points: torch.Tensor,
    selected_indices: torch.Tensor,
    attn: torch.Tensor,
) -> torch.Tensor:
    """Interpolate canonical neighbour positions with the deformed attention weights.

    The ablation baseline. Retained to reproduce the comparison figure; it is not the released
    default because sparse attention makes it collapse rays onto individual support points.
    """
    k = selected_indices.shape[-1]
    weights = _normalise_weights(attn, k).unsqueeze(-1)
    return (weights * canonical_points[selected_indices]).sum(dim=-2)


def transfer(method: str, **kwargs: object) -> torch.Tensor:
    """Dispatch by name, matching the ``--canonical_method`` flag of the archived render driver."""
    if method == "deformation_aware":
        return deformation_aware_transfer(**kwargs)  # type: ignore[arg-type]
    if method == "naive":
        allowed = {"canonical_points", "selected_indices", "attn"}
        return naive_transfer(**{k: v for k, v in kwargs.items() if k in allowed})  # type: ignore[arg-type]
    raise ValueError(f"unknown transfer method {method!r}; expected one of {TRANSFER_METHODS}")

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

#: Highest band implemented. Raising this requires new constants *and* a checkpoint audit: the
#: coefficient tensor width is ``(degree + 1) ** 2``, which is part of the stored parameter shape.
MAX_SH_DEGREE = 3

#: Band normalisation constants, indexed within their band. Full double precision: truncating them
#: by a digit or two is a 1e-16 relative difference, far below float32 resolution on the render
#: path.
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)

__all__ = [
    "MAX_SH_DEGREE",
    "SH_C0",
    "SH_C1",
    "SH_C2",
    "SH_C3",
    "active_sh_degree",
    "num_sh_bases",
    "sh_basis",
    "spherical_harmonics",
]


def _check_degree(degree: int, name: str = "degree") -> None:
    if not isinstance(degree, int):
        raise TypeError(f"{name} must be an int, got {type(degree).__name__}")
    if not 0 <= degree <= MAX_SH_DEGREE:
        raise ValueError(f"{name} must be in [0, {MAX_SH_DEGREE}], got {degree}")


def num_sh_bases(degree: int) -> int:
    """Number of basis functions up to and including ``degree``."""
    _check_degree(degree)
    return (degree + 1) ** 2


def active_sh_degree(step: int, interval: float, max_degree: int = MAX_SH_DEGREE) -> int:
    """Degree unlocked at training ``step``: one further band every ``interval`` steps.

    ``interval`` is a config value that appears both as a step count (``10000``) and as a fraction
    (``0.1``, i.e. everything unlocked immediately), hence the float type.
    """
    if not isinstance(step, int):
        raise TypeError(f"step must be an int, got {type(step).__name__}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if interval <= 0:
        raise ValueError(f"sh_degree_interval must be positive, got {interval}")
    _check_degree(max_degree, "max_degree")
    return min(int(step // interval), max_degree)


def sh_basis(degree: int, dirs: Tensor) -> Tensor:
    """Evaluate the real SH basis at ``dirs``.

    Args:
        degree: 0 to :data:`MAX_SH_DEGREE`.
        dirs: ``(..., 3)`` directions over any leading batch shape. Expected to be unit length;
            they are not normalised here, so callers holding raw offsets should use
            :func:`spherical_harmonics` or normalise first.

    Returns:
        ``(..., (degree + 1) ** 2)``, dtype and device of ``dirs``.
    """
    _check_degree(degree)
    if dirs.shape[-1] != 3:
        raise ValueError(f"dirs must end in a dimension of 3, got {tuple(dirs.shape)}")

    x, y, z = dirs.unbind(-1)
    terms = [torch.full_like(x, SH_C0)]

    if degree >= 1:
        terms += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]

        if degree >= 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            terms += [
                SH_C2[0] * xy,
                SH_C2[1] * yz,
                SH_C2[2] * (2.0 * zz - xx - yy),
                SH_C2[3] * xz,
                SH_C2[4] * (xx - yy),
            ]

            if degree >= 3:
                terms += [
                    SH_C3[0] * y * (3.0 * xx - yy),
                    SH_C3[1] * xy * z,
                    SH_C3[2] * y * (4.0 * zz - xx - yy),
                    SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
                    SH_C3[4] * x * (4.0 * zz - xx - yy),
                    SH_C3[5] * z * (xx - yy),
                    SH_C3[6] * x * (xx - 3.0 * yy),
                ]

    return torch.stack(terms, dim=-1)


def spherical_harmonics(degree: int, dirs: Tensor, coeffs: Tensor) -> Tensor:
    """Expand per-element SH coefficients along ``dirs``.

    Args:
        degree: the *active* degree. Coefficients in bands above it are excluded from the sum, which
            is what makes the ``sh_degree_interval`` ramp work; see :func:`active_sh_degree`.
        dirs: ``(..., 3)``, normalised internally because the render path passes raw
            point-minus-origin offsets.
        coeffs: ``(..., K, D)`` with the same leading batch shape as ``dirs`` and
            ``K >= (degree + 1) ** 2``. ``D`` is the feature width, not necessarily 3 -- the
            attention values these feed are 64-wide.

    Returns:
        ``(..., D)``.
    """
    bases_needed = num_sh_bases(degree)
    if dirs.shape[-1] != 3:
        raise ValueError(f"dirs must end in a dimension of 3, got {tuple(dirs.shape)}")
    batch_shape = dirs.shape[:-1]
    if coeffs.shape[:-2] != batch_shape:
        raise ValueError(
            "coeffs must share the batch shape of dirs and add (K, D): expected "
            f"{tuple(batch_shape)} + (K, D), got {tuple(coeffs.shape)}"
        )
    if coeffs.shape[-2] < bases_needed:
        raise ValueError(
            f"degree {degree} needs {bases_needed} bases, coeffs provides {coeffs.shape[-2]}"
        )

    bases = sh_basis(degree, F.normalize(dirs, p=2, dim=-1))
    return (bases[..., None] * coeffs[..., :bases_needed, :]).sum(dim=-2)

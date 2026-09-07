from __future__ import annotations

import re

import torch
from torch import Tensor, nn

#: Negative slope for ``leakyrelu``, kept at 0.2 rather than torch's default 0.01: every archived
#: LeakyReLU network was trained at 0.2.
LEAKY_RELU_NEGATIVE_SLOPE = 0.2

#: Accepted spellings, lowercased. ``softplus`` additionally takes its three coefficients in the
#: name itself (e.g. ``softplus_1_1_0``), which is why it is not a plain string match.
SUPPORTED_ACTIVATIONS: tuple[str, ...] = (
    "none",
    "relu",
    "leakyrelu",
    "sigmoid",
    "silu",
    "relu+1",
    "softplus_<c1>_<c2>_<c3>",
)

_SOFTPLUS_PATTERN = re.compile(r"^softplus_(?P<c1>[^_]+)_(?P<c2>[^_]+)_(?P<c3>[^_]+)$")


class UnknownActivationError(ValueError):
    """Raised for an activation name this release does not implement."""


class PlusOne(nn.Module):
    """Shift by one, keeping a preceding ReLU's output strictly positive."""

    def forward(self, x: Tensor) -> Tensor:
        return x + 1


class ScaledSoftplus(nn.Module):
    """``c1 * softplus(c2 * x + c3)``.

    The affine wrapper is what the ``softplus_<c1>_<c2>_<c3>`` config spelling encodes; the shipped
    configs all use ``softplus_1_1_0``, which is plain softplus, but the coefficients are read from
    the name so a config that changes them is not silently ignored.
    """

    def __init__(self, c1: float, c2: float, c3: float) -> None:
        super().__init__()
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = float(c3)

    def forward(self, x: Tensor) -> Tensor:
        return self.c1 * torch.nn.functional.softplus(self.c2 * x + self.c3)

    def extra_repr(self) -> str:
        return f"c1={self.c1}, c2={self.c2}, c3={self.c3}"


def _parse_softplus(name: str) -> ScaledSoftplus:
    match = _SOFTPLUS_PATTERN.match(name)
    if match is None:
        raise UnknownActivationError(
            f"activation {name!r} looks like a softplus but does not match "
            "'softplus_<c1>_<c2>_<c3>' (e.g. 'softplus_1_1_0')"
        )
    try:
        coefficients = [float(match.group(key)) for key in ("c1", "c2", "c3")]
    except ValueError as exc:
        raise UnknownActivationError(
            f"activation {name!r} has non-numeric softplus coefficients"
        ) from exc
    return ScaledSoftplus(*coefficients)


def make_activation(name: str | None, *, inplace: bool = True) -> nn.Module:
    """Build the activation module a config string names.

    ``None`` and ``"none"`` (in any casing) give :class:`torch.nn.Identity`, so a network can be
    built with its nonlinearity switched off without the caller special-casing it.

    ``inplace`` defaults to true: these activations sit directly on the output of a linear layer,
    where reusing the buffer is safe and the saved allocation is material at full render
    resolution. Pass ``inplace=False`` when the input is a tensor the caller still needs — notably
    a leaf that requires grad, which torch refuses to overwrite.

    Raises:
        UnknownActivationError: if the name is not one this release implements.
    """
    if name is None:
        return nn.Identity()
    if not isinstance(name, str):
        raise UnknownActivationError(f"activation name must be a string or None, got {name!r}")

    key = name.strip().lower()
    if key.startswith("softplus"):
        return _parse_softplus(key)

    if key == "none":
        return nn.Identity()
    if key == "relu":
        return nn.ReLU(inplace)
    if key == "leakyrelu":
        return nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE, inplace)
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "silu":
        return nn.SiLU(inplace)
    if key == "relu+1":
        return nn.Sequential(nn.ReLU(inplace), PlusOne())

    raise UnknownActivationError(
        f"activation {name!r} is not implemented in this release; supported names are "
        f"{', '.join(SUPPORTED_ACTIVATIONS)} (case-insensitive), or None"
    )

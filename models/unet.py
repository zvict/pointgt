from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.activations import make_activation

#: Channel width of each encoder level. Fixed by the archived weight shapes, not configurable: the
#: ``channel_factor`` knob that scaled them is 1 in every shipped config.
LEVEL_WIDTHS: tuple[int, int, int] = (128, 256, 512)

#: Config knobs whose archived variants were dropped, mapped to the one value each may still take.
_FIXED_OPTIONS: dict[str, Any] = {
    "affine_layer": -1,
    "bilinear": False,
    "channel_factor": 1,
    "group_last": False,
    "groups": 1,
    "norm": "none",
    "single": True,
    "use_outc": True,
}


class SingleConv(nn.Module):
    """A 3x3 convolution followed by an activation.

    ``double_conv`` is the contracted attribute name; index 0 is the convolution and index 1 the
    parameter-free activation, which is why the archived keys stop at ``double_conv.0``.
    """

    def __init__(self, in_channels: int, out_channels: int, act: str = "relu") -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            make_activation(act),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Halve the resolution by max-pooling, then convolve."""

    def __init__(self, in_channels: int, out_channels: int, act: str = "relu") -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SingleConv(in_channels, out_channels, act=act),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsample by transposed convolution, concatenate the skip map, then convolve.

    ``up`` halves the channel count while doubling the resolution, so concatenating the skip map
    brings the width back to ``in_channels`` for ``conv``.
    """

    def __init__(self, in_channels: int, out_channels: int, act: str = "relu") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = SingleConv(in_channels, out_channels, act=act)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        diff_y = skip.shape[-2] - x.shape[-2]
        diff_x = skip.shape[-1] - x.shape[-1]
        if diff_x or diff_y:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    """1x1 projection from the finest level onto the output channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Three-level U-Net decoding a per-pixel feature map to RGB.

    ``in_channels`` is config-derived rather than fixed: it is the width of the encoded value
    features, doubled upstream when ``models.unet.double_channel`` is set. Both archived families
    land on 128 by different routes, which is why the first convolution has one shape across all
    477 checkpoints even though the configs feeding it differ.

    ``inp_scale`` multiplies the input before the first convolution, letting a config lift a
    small-magnitude feature map into a range the convolutions were tuned for.
    """

    #: Two 2x max-pools, so anything smaller collapses the coarsest level to zero pixels.
    MIN_SPATIAL_SIZE = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        *,
        act: str = "relu",
        last_act: str = "none",
        inp_scale: float = 1.0,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels < 1:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.inp_scale = float(inp_scale)
        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype

        fine, mid, coarse = LEVEL_WIDTHS
        self.inc = SingleConv(self.in_channels, fine, act=act)
        self.down1 = Down(fine, mid, act=act)
        self.down2 = Down(mid, coarse, act=act)
        self.up1 = Up(coarse, mid, act=act)
        self.up2 = Up(mid, fine, act=act)
        self.outc = OutConv(fine, self.out_channels)
        self.last_act = make_activation(last_act)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"UNet expects a (N, C, H, W) feature map, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"UNet expects {self.in_channels} channels, got {x.shape[1]}")
        if min(x.shape[-2:]) < self.MIN_SPATIAL_SIZE:
            raise ValueError(
                f"UNet needs at least {self.MIN_SPATIAL_SIZE}x{self.MIN_SPATIAL_SIZE} pixels, "
                f"got {tuple(x.shape[-2:])}"
            )

        context = (
            torch.autocast(device_type=x.device.type, dtype=self.amp_dtype)
            if self.use_amp
            else nullcontext()
        )
        with context:
            x = x * self.inp_scale
            fine = self.inc(x)
            mid = self.down1(fine)
            coarse = self.down2(mid)
            out = self.up1(coarse, mid)
            out = self.up2(out, fine)
            return self.last_act(self.outc(out))


def unet_from_config(
    config: Mapping[str, Any],
    in_channels: int,
    *,
    out_channels: int = 3,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> UNet:
    """Build a :class:`UNet` from a ``models.unet`` config block.

    Any knob whose archived variant was dropped raises instead of being ignored, because the closest
    supported network would have different weight shapes and would load an archived checkpoint into
    the wrong graph. ``use`` and ``double_channel`` stay the caller's concern: the first decides
    whether a decoder exists at all, the second is folded into ``in_channels`` before it gets here.
    """
    for key, fixed in _FIXED_OPTIONS.items():
        if key in config and config[key] != fixed:
            raise ValueError(
                f"models.unet.{key}={config[key]!r} is not part of the release; every archived "
                f"checkpoint was trained with {key}={fixed!r}"
            )
    return UNet(
        in_channels,
        out_channels,
        act=str(config.get("act", "relu")),
        last_act=str(config.get("last_act", "none")),
        inp_scale=float(config.get("inp_scale", 1.0)),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

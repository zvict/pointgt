from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import lpips
import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "AdaptiveBerHuLoss",
    "LPIPSLoss",
    "RenderingLoss",
    "SSIMLoss",
    "build_loss",
    "supported_losses",
]


def _validate_pair(pred: Tensor, target: Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if pred.ndim != 4:
        raise ValueError(f"expected channels-last (B, H, W, C) tensors, got ndim={pred.ndim}")


def _to_nchw(image: Tensor) -> Tensor:
    return image.permute(0, 3, 1, 2)


class AdaptiveBerHuLoss(nn.Module):
    """Reverse Huber: L1 on small residuals, L2 on large ones, switching at a data-driven point.

    The switch sits at ``fraction`` of the largest residual in the batch, so the L2 branch keeps
    biting as the render improves instead of the whole image sliding under a fixed threshold.
    """

    def __init__(self, fraction: float = 0.2) -> None:
        super().__init__()
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must lie in (0, 1], got {fraction}")
        self.fraction = float(fraction)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        residual = pred - target
        magnitude = residual.abs()
        threshold = magnitude.max().detach() * self.fraction
        threshold = threshold.clamp_min(torch.finfo(magnitude.dtype).tiny)
        quadratic = (residual * residual + threshold * threshold) / (2 * threshold)
        return torch.where(magnitude < threshold, magnitude, quadratic).mean()

    def extra_repr(self) -> str:
        return f"fraction={self.fraction:g}"


class SSIMLoss(nn.Module):
    """``1 - SSIM`` using this repository's own training kernel, not the one the metrics module
    reports.

    This release deliberately contains two different SSIMs.

    * **Here (training).** An 11x11 Gaussian with sigma 1.5 convolved in ``same`` mode with zero
      padding, and a non-standard ``+1e-6`` added to the denominator -- this is the objective the
      released checkpoints were trained against.
    * **``utils/metrics.py`` (reporting).** ``piq.ssim``, which convolves in ``valid`` mode. This
      is what the evaluation aggregated into the paper's tables.

    They disagree, and the gap is largest at small patch sizes: on a 16x16 patch, ``valid``
    convolution leaves a 6x6 SSIM map, so border pixels receive almost no structural gradient.
    Do not "unify" them. Changing this class changes the training objective; changing the metrics
    one changes what the numbers mean.

    No output clamp. Every shipped config sets ``models.last_act: none``, so a render legitimately
    leaves ``[0, 1]`` mid-training, and this kernel accepts it. ``piq`` would not, which is the only
    reason a clamp ever existed here.
    """

    def __init__(self, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        if data_range <= 0:
            raise ValueError(f"data_range must be positive, got {data_range}")
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.register_buffer("_window", self._make_window(self.window_size, self.sigma), persistent=False)

    @staticmethod
    def _make_window(window_size: int, sigma: float) -> Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window_2d = gauss[:, None] @ gauss[None, :]
        return window_2d[None, None]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        img1 = _to_nchw(pred)
        img2 = _to_nchw(target)
        channel = img1.shape[1]
        window = self._window.to(dtype=img1.dtype, device=img1.device).expand(channel, 1, -1, -1)
        pad = self.window_size // 2

        mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-6
        )
        return 1.0 - ssim_map.mean()

    def extra_repr(self) -> str:
        return f"data_range={self.data_range:g}, window_size={self.window_size}, sigma={self.sigma:g}"


class LPIPSLoss(nn.Module):
    """Perceptual distance from the pip ``lpips`` package, reduced to a scalar.

    ``normalize`` follows the upstream meaning: ``False`` treats the input as already living in
    ``[-1, 1]``. The archived runs passed ``[0, 1]`` images with that default, so the effective
    input range was half of what the network was calibrated for. Preserving it is what makes a
    retrain optimise the same objective the published checkpoints were trained against; set
    ``normalize=True`` for the scaling upstream intends.
    """

    def __init__(self, net: str = "vgg", *, normalize: bool = False) -> None:
        super().__init__()
        if net not in {"alex", "squeeze", "vgg"}:
            raise ValueError(f"unknown LPIPS backbone {net!r}; expected alex, squeeze or vgg")
        self.backbone = net
        self.normalize = bool(normalize)
        self.net = lpips.LPIPS(net=net, verbose=False)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        self.net.eval()

    def train(self, mode: bool = True) -> LPIPSLoss:
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        distance = self.net(_to_nchw(pred), _to_nchw(target), normalize=self.normalize)
        return distance.mean()

    def extra_repr(self) -> str:
        return f"net={self.backbone}, normalize={self.normalize}"


#: Builders for the terms the released configs weight. Keys are the archived config names.
_TERM_FACTORIES: Mapping[str, Callable[[], nn.Module]] = MappingProxyType(
    {
        "mse": nn.MSELoss,
        "lpips": lambda: LPIPSLoss("vgg"),
        "ssim": SSIMLoss,
        "ada_berhu": AdaptiveBerHuLoss,
    }
)

#: Names the archived blocks carry but never weight. Recognised so those blocks still load; a
#: non-zero weight raises, because no published result was produced with the term switched on.
_RETIRED_TERMS = frozenset({"lpips_alex"})


def supported_losses() -> tuple[str, ...]:
    """The loss names this release can actually build, in a stable order."""
    return tuple(sorted(_TERM_FACTORIES))


class RenderingLoss(nn.Module):
    """Sum of the weighted rendering terms.

    Zero-weighted terms are dropped at construction rather than multiplied by zero at every step,
    so a config that disables LPIPS never pays for the perceptual forward pass.
    """

    def __init__(self, weights: Mapping[str, float]) -> None:
        super().__init__()
        resolved = _resolve_weights(weights)
        if not resolved:
            raise ValueError("every loss weight is zero; there would be nothing to optimise")
        self.terms = nn.ModuleDict({name: _TERM_FACTORIES[name]() for name in resolved})
        self._weights = resolved

    @property
    def weights(self) -> Mapping[str, float]:
        """The active terms and their weights, in config order."""
        return MappingProxyType(self._weights)

    def breakdown(self, pred: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Per-term *weighted* contributions, for logging what each term actually pushed."""
        _validate_pair(pred, target)
        return {name: self._weights[name] * term(pred, target) for name, term in self.terms.items()}

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        _validate_pair(pred, target)
        values = [self._weights[name] * term(pred, target) for name, term in self.terms.items()]
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total

    def __repr__(self) -> str:
        terms = ", ".join(f"{name}={weight:g}" for name, weight in self._weights.items())
        return f"{type(self).__name__}({terms})"


def _resolve_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Validate every entry, then keep only the terms that carry weight."""
    resolved: dict[str, float] = {}
    for name, raw in weights.items():
        if name not in _TERM_FACTORIES and name not in _RETIRED_TERMS:
            raise ValueError(
                f"unknown loss {name!r}; this release ships {', '.join(supported_losses())}"
            )
        try:
            weight = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"loss weight for {name!r} is not a number: {raw!r}") from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"loss weight for {name!r} must be finite and >= 0, got {raw!r}")
        if weight == 0.0:
            continue
        if name in _RETIRED_TERMS:
            raise ValueError(
                f"loss {name!r} is weighted {weight:g}, but no released run ever weighted it; "
                "this release ships it at 0 only"
            )
        resolved[name] = weight
    return resolved


def build_loss(config: Mapping[str, Any]) -> RenderingLoss:
    """Build the rendering objective from a ``training.losses`` weight mapping.

    Pass the weight mapping itself (``cfg.training.losses``), not the whole config: taking the
    whole config would make a typo in the block's location look like an empty objective.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"expected a mapping of loss name to weight, got {type(config).__name__}")
    for marker in ("training", "losses"):
        if marker in config:
            raise ValueError(
                f"build_loss takes the loss-weight mapping itself, e.g. cfg.training.losses; "
                f"got a mapping containing {marker!r}"
            )
    return RenderingLoss(config)

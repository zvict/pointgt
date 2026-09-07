from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import piq
import torch

if TYPE_CHECKING:
    from torch import nn

#: Version tag on the sidecar, so a reader can tell which layout it is holding.
METRICS_SCHEMA = "pointgt.metrics/1"

#: Sidecar filename. Fixed, because tooling globs for it.
METRICS_FILENAME = "metrics.json"

#: Perceptual backbones the ``lpips`` package ships. There is deliberately no default: the paper's
#: table mixes Alex and VGG between rows and the release has to state which one produced a number.
LPIPS_BACKBONES = frozenset({"alex", "vgg", "squeeze"})

#: Linear-head weight revision. ``0.1`` is what the archived runs used and what the table reports.
LPIPS_VERSION = "0.1"

_MARKER_NUMBER = r"[-+]?(?:\d+\.\d+|\d+|nan|inf)"
_MARKER_RE = re.compile(
    rf"^a-(?P<head>PHYS-)?PSNR-(?P<psnr>{_MARKER_NUMBER})"
    rf"-SSIM-(?P<ssim>{_MARKER_NUMBER})"
    rf"-LPIPSv-(?P<lpips_vgg>{_MARKER_NUMBER})"
    rf"(?:-IntPSNR-(?P<int_psnr>{_MARKER_NUMBER}))?$"
)


def _as_metric_input(name: str, image: torch.Tensor, data_range: float) -> torch.Tensor:
    """Validate one image batch and put it in the layout and precision the metrics assume."""
    if not torch.is_tensor(image):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(image).__name__}")
    if image.ndim != 4:
        raise ValueError(f"{name} must be (N, C, H, W), got shape {tuple(image.shape)}")
    if image.shape[1] not in (1, 3):
        raise ValueError(
            f"{name} must be channels-first with 1 or 3 channels, got shape {tuple(image.shape)}"
            " -- pass (N, C, H, W), not (N, H, W, C)"
        )
    if not image.is_floating_point():
        raise TypeError(f"{name} must be a float tensor, got dtype {image.dtype}")

    out = image.detach().to(dtype=torch.float32)

    if not torch.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values; metrics would be meaningless")
    low, high = float(out.min()), float(out.max())
    if low < 0.0 or high > data_range:
        raise ValueError(
            f"{name} must lie in [0, {data_range}], got [{low:.4g}, {high:.4g}]"
            " -- clamp the render before scoring it"
        )
    return out


def _validate_pair(
    pred: torch.Tensor, target: torch.Tensor, data_range: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    x = _as_metric_input("pred", pred, data_range)
    y = _as_metric_input("target", target, data_range)
    if x.shape != y.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    return x, y


def psnr(pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    """Per-frame PSNR in dB. Returns shape ``(N,)``.

    ``piq`` floors the MSE with ``EPS = 1e-8`` before the log, so identical frames score exactly
    80.0 dB rather than ``+inf``. That ceiling is load-bearing in two ways: the sidecar stays valid
    JSON (which has no infinity), and at a realistic MSE around 1e-3 the floor shifts the result by
    under 1e-4 dB, far below the parity tolerance in the release-provenance notes kept outside this repository.
    """
    x, y = _validate_pair(pred, target, data_range)
    return piq.psnr(x, y, data_range=data_range, reduction="none")


def ssim(pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    """Per-frame SSIM. Returns shape ``(N,)``.

    Left at ``piq``'s defaults on purpose -- kernel 11 / sigma 1.5 / ``downsample=True`` is what
    produced every archived SSIM, and the downsampling factor depends on the image size, so a
    re-scored render only matches at the resolution it was evaluated at.
    """
    x, y = _validate_pair(pred, target, data_range)
    if min(x.shape[-2:]) < 11:
        raise ValueError(
            "SSIM needs frames of at least 11x11 for the Gaussian kernel, got "
            f"{tuple(x.shape[-2:])}"
        )
    return piq.ssim(x, y, data_range=data_range, reduction="none")


class LpipsMetric:
    """Per-frame LPIPS distance from the pip ``lpips`` package.

    ``backbone`` is required. The paper's table quotes VGG on some rows and Alex on others, and the
    two are not on the same scale, so a number without its backbone recorded is unusable.

    ``normalize`` selects the input convention and defaults to the archived one. ``lpips`` expects
    ``[-1, 1]`` when ``normalize=False``; passing ``[0, 1]`` renders straight in with the flag left
    alone means the internal scaling layer sees half the intended range. Every published LPIPS
    number carries that, which is why the release reproduces it by default instead of quietly
    correcting it -- ``normalize=True`` gives the textbook value and a different number, so it must
    be an opt-in that the sidecar records.

    The backbone is built lazily: importing this module should not pull in torchvision or touch the
    weight cache, so PSNR and SSIM stay usable where the perceptual weights are unavailable.
    """

    __slots__ = ("_device", "_model", "backbone", "normalize", "version")

    def __init__(
        self,
        backbone: str,
        *,
        version: str = LPIPS_VERSION,
        normalize: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        if backbone not in LPIPS_BACKBONES:
            raise ValueError(
                f"unknown LPIPS backbone {backbone!r}; expected one of {sorted(LPIPS_BACKBONES)}"
            )
        self.backbone = backbone
        self.version = version
        self.normalize = normalize
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        self._model: nn.Module | None = None

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            import lpips as _lpips

            model = _lpips.LPIPS(net=self.backbone, version=self.version, verbose=False)
            self._model = model.eval().to(self._device)
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)
        return self._model

    def to(self, device: torch.device | str) -> LpipsMetric:
        self._device = torch.device(device)
        if self._model is not None:
            self._model = self._model.to(self._device)
        return self

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0
    ) -> torch.Tensor:
        """Per-frame LPIPS distance. Returns shape ``(N,)``."""
        x, y = _validate_pair(pred, target, data_range)
        if data_range != 1.0:
            x, y = x / data_range, y / data_range
        model = self.model
        with torch.no_grad():
            distance = model(x.to(self._device), y.to(self._device), normalize=self.normalize)
        return distance.reshape(x.shape[0]).cpu()

    def describe(self) -> dict[str, Any]:
        """The configuration to record beside the numbers it produced."""
        return {"backbone": self.backbone, "version": self.version, "normalize": self.normalize}


@dataclass(frozen=True)
class FrameMetrics:
    """One test frame's scores. ``lpips`` is ``None`` when no backbone was configured."""

    index: int
    psnr: float
    ssim: float
    lpips: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"index": self.index, "psnr": self.psnr, "ssim": self.ssim}
        if self.lpips is not None:
            out["lpips"] = self.lpips
        return out


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


class MetricAggregator:
    """Accumulates per-frame scores over an evaluation pass and writes the sidecar.

    Frames are indexed in arrival order, matching the test loader, so ``per_frame`` in the sidecar
    lines up with the rendered filenames.
    """

    __slots__ = ("_data_range", "_frames", "_lpips")

    def __init__(self, *, lpips_metric: LpipsMetric | None = None, data_range: float = 1.0) -> None:
        if data_range <= 0:
            raise ValueError(f"data_range must be positive, got {data_range}")
        self._lpips = lpips_metric
        self._data_range = float(data_range)
        self._frames: list[FrameMetrics] = []

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> tuple[FrameMetrics, ...]:
        return tuple(self._frames)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[FrameMetrics, ...]:
        """Score a batch of frames and append them. Returns just the frames added by this call."""
        psnr_values = psnr(pred, target, data_range=self._data_range)
        ssim_values = ssim(pred, target, data_range=self._data_range)
        if self._lpips is not None:
            lpips_values: list[float | None] = [
                float(v) for v in self._lpips(pred, target, data_range=self._data_range)
            ]
        else:
            lpips_values = [None] * psnr_values.shape[0]

        start = len(self._frames)
        added = tuple(
            FrameMetrics(
                index=start + offset,
                psnr=float(psnr_values[offset]),
                ssim=float(ssim_values[offset]),
                lpips=lpips_values[offset],
            )
            for offset in range(psnr_values.shape[0])
        )
        self._frames.extend(added)
        return added

    def summary(self) -> dict[str, Any]:
        """The full record: aggregate, LPIPS configuration, and the per-frame series."""
        if not self._frames:
            raise ValueError("no frames accumulated; call update() before summary()")

        aggregate: dict[str, float] = {
            "psnr": _mean([f.psnr for f in self._frames]),
            "ssim": _mean([f.ssim for f in self._frames]),
        }
        scored = [f.lpips for f in self._frames if f.lpips is not None]
        if scored:
            if len(scored) != len(self._frames):
                raise ValueError("LPIPS is present on some frames but not all; the mean would lie")
            aggregate["lpips"] = _mean(scored)

        return {
            "schema": METRICS_SCHEMA,
            "n_frames": len(self._frames),
            "aggregate": aggregate,
            "lpips": self._lpips.describe() if self._lpips is not None else None,
            "per_frame": [f.as_dict() for f in self._frames],
            "legacy_marker": self.legacy_marker(),
        }

    def legacy_marker(self) -> str | None:
        """The archived marker-directory name, for cross-checking against a recovered run.

        Emitted only for VGG. The marker's field is literally named ``LPIPSv``, so writing an Alex
        distance into it would reproduce the mixed-backbone error the published table already has.
        """
        if self._lpips is None or self._lpips.backbone != "vgg":
            return None
        aggregate = {
            "psnr": _mean([f.psnr for f in self._frames]),
            "ssim": _mean([f.ssim for f in self._frames]),
            "lpips": _mean([f.lpips for f in self._frames if f.lpips is not None]),
        }
        return (
            f"a-PSNR-{aggregate['psnr']:.4f}"
            f"-SSIM-{aggregate['ssim']:.4f}"
            f"-LPIPSv-{aggregate['lpips']:.4f}"
            f"-IntPSNR-{0.0:.4f}"
        )

    def write_json(self, path: str | Path) -> Path:
        """Write the sidecar. A ``.json`` path is used as-is; anything else is a directory."""
        target = Path(path)
        destination = target if target.suffix == ".json" else target / METRICS_FILENAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination


def read_metrics_json(path: str | Path) -> dict[str, Any]:
    """Read a sidecar, rejecting a layout this version does not understand."""
    target = Path(path)
    if target.is_dir():
        target = target / METRICS_FILENAME
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{target} does not hold a metrics object")
    schema = payload.get("schema")
    if schema != METRICS_SCHEMA:
        raise ValueError(f"{target} has schema {schema!r}, expected {METRICS_SCHEMA!r}")
    return payload


def parse_legacy_marker(name: str) -> dict[str, Any]:
    """Recover the numbers from an archived marker-directory name.

    Accepts the main-head form and the ``a-PHYS-`` auxiliary-head form, whose string omits
    ``IntPSNR`` entirely. Raises on anything else rather than returning partial numbers, because a
    truncated path was exactly how these records used to be lost without anyone noticing.
    """
    match = _MARKER_RE.match(Path(name).name)
    if match is None:
        raise ValueError(f"not a PointGT metrics marker: {name!r}")
    int_psnr = match.group("int_psnr")
    return {
        "head": "phys" if match.group("head") else "main",
        "psnr": float(match.group("psnr")),
        "ssim": float(match.group("ssim")),
        "lpips_vgg": float(match.group("lpips_vgg")),
        "int_psnr": float(int_psnr) if int_psnr is not None else None,
    }

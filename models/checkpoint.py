from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

#: Tensors whose leading dimension is the point count. Their size is a property of the trained
#: checkpoint, not of the config, so they are replaced wholesale on load.
DYNAMIC_PARAMETERS = frozenset(
    {
        "points",
        "pc_feats",
        "points_influ_scores",
        "points_scaler",
        "points_density",
        "points_normals",
        "points_last_grad",
        "points_acc_grad",
        "points_acc_grad_norm",
        "points_grad_cnt",
    }
)

#: Absent from the ``nvs_ours`` family and reconstructible from config or point count. Everything
#: else that is missing is an error.
DERIVABLE_MISSING = frozenset({"points_scaler", "points_density", "points_normals", "bkg_score"})

#: Present in every archived family (the release-provenance notes kept outside this repository). The four
#: ``points_*grad*`` entries are densification bookkeeping rather than learned weights, but a
#: checkpoint missing them fails to load correctly. Checked on save, so the save and load sides of
#: this module cannot drift apart silently.
REQUIRED_ON_SAVE = frozenset(
    {
        "points",
        "pc_feats",
        "points_influ_scores",
        "points_last_grad",
        "points_acc_grad",
        "points_acc_grad_norm",
        "points_grad_cnt",
        "bkg_feats",
        "bkg_scaler",
        "proximity_attn.score_scale",
    }
)


@dataclass(frozen=True)
class LoadReport:
    """The audited outcome of a checkpoint load."""

    path: str
    sha256: str
    source_schema: str
    step: int
    n_points: int
    migrations: tuple[str, ...]
    derived: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[str, ...]
    unequal: tuple[str, ...]

    @property
    def exact(self) -> bool:
        return not (self.missing or self.unexpected or self.shape_mismatch or self.unequal)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"exact": self.exact}


def sha256_file(path: str | Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint(
    path: str | Path, *, expected_step: int | None = None, require_step: bool = True
) -> tuple[dict[str, torch.Tensor], int, str]:
    """Unwrap the two archived containers.

    ``model.save`` wrote ``{"<step>": state_dict}``; ``train.py`` separately wrote
    ``model_<step>.pth`` holding a bare ``state_dict``. The wrapped form is recognised by its key
    being an integer label over a mapping, rather than by key count alone, which would misread a
    single-tensor state_dict.

    A bare container carries its step only in its filename. ``require_step=False`` accepts one
    whose name does not encode a step and reports ``-1``: a fine-tune restarts the counter at zero,
    so the source step is genuinely unused there. ``--resume`` leaves it True, because resuming to
    the wrong step would silently mis-schedule every warmup and decay.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"Unsupported checkpoint container: {path}")

    if len(payload) == 1:
        label, inner = next(iter(payload.items()))
        if str(label).isdigit() and isinstance(inner, Mapping):
            step = int(label)
            if expected_step is not None and step != expected_step:
                raise ValueError(f"Checkpoint step is {step}, expected {expected_step}")
            return dict(inner), step, "wrapped"

    if all(torch.is_tensor(v) for v in payload.values()):
        step = expected_step
        if step is None:
            suffix = Path(path).stem.rsplit("_", 1)[-1]
            if suffix.isdigit():
                step = int(suffix)
            elif require_step:
                raise ValueError(
                    f"A bare checkpoint needs expected_step or a model_<step>.pth name: {path}"
                )
            else:
                step = -1
        return dict(payload), int(step), "bare"

    raise ValueError(f"Unsupported checkpoint container: {path}")


def extra_path(checkpoint_path: str | Path, name: str) -> Path:
    """Sibling file holding a non-model payload, e.g. ``optimizers`` -> ``<dir>/optimizers.pth``.

    ``model.pth`` sits next to ``optimizers.pth``, ``schedulers.pth`` and ``scaler.pth`` in one
    directory. Keeping the convention in one function stops the reader and the writer from
    disagreeing about the filename.
    """
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Invalid sidecar name: {name!r}")
    return Path(checkpoint_path).parent / f"{name}.pth"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    step: int,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write ``{"<step>": state_dict}`` -- the archived container -- plus any sidecar payloads.

    The model file holds exactly one key, the step as a decimal string, mapping to the full
    ``state_dict``; optimiser, scheduler and ``GradScaler`` states go to siblings named by
    :func:`extra_path`. Keeping those out of the model file is what lets :func:`read_checkpoint`
    recognise the wrapped container by its shape rather than by counting keys.

    Tensors are written on whatever device they live on.

    Returns the paths written, keyed ``"model"`` plus one entry per ``extra`` name.
    """
    step_int = int(step)
    if step_int != step or step_int < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")

    state = model.state_dict()
    absent = sorted(REQUIRED_ON_SAVE - set(state))
    if absent:
        raise ValueError(
            f"Refusing to write a checkpoint missing contract keys {absent}; the loader "
            f"indexes them unguarded and requires them"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({str(step_int): state}, path)

    written = {"model": path}
    for name, payload in (extra or {}).items():
        target = extra_path(path, name)
        torch.save(payload, target)
        written[name] = target
    return written


def _migrate(state: dict[str, torch.Tensor], current: Mapping[str, torch.Tensor]) -> list[str]:
    """Apply named, lossless shape migrations. Anything unrecognised is left to fail loudly."""
    done: list[str] = []
    key = "append_bkg_points_embedv"
    if key in state and key in current:
        want, have = current[key].shape, state[key].shape
        if len(have) == 1 and len(want) == 2 and want[0] == 1 and want[1] == have[0]:
            state[key] = state[key].unsqueeze(0)
            done.append(f"{key}: {tuple(have)} -> {tuple(state[key].shape)}")
    return done


def _materialize_dynamic(model: nn.Module, state: Mapping[str, torch.Tensor]) -> list[str]:
    """Re-assign per-point parameters from the checkpoint, and derive the ones it omits."""
    if "points" not in state:
        raise ValueError("Checkpoint has no 'points' tensor; not a PointGT checkpoint")

    device = model.points.device if hasattr(model, "points") else torch.device("cpu")
    existing = dict(model.named_parameters())
    for name in sorted(DYNAMIC_PARAMETERS & set(state)):
        if not hasattr(model, name):
            continue
        previous = existing.get(name)
        requires_grad = bool(previous.requires_grad) if previous is not None else False
        tensor = state[name].detach().to(device).clone()
        setattr(model, name, nn.Parameter(tensor, requires_grad=requires_grad))

    count = int(state["points"].shape[0])
    dtype = state["points"].dtype
    derived: list[str] = []
    if "points_scaler" not in state and hasattr(model, "points_scaler"):
        model.points_scaler = nn.Parameter(
            torch.ones((count, 1), device=device, dtype=dtype),
            requires_grad=bool(model.points_scaler.requires_grad),
        )
        derived.append("points_scaler=ones")
    if "points_density" not in state and hasattr(model, "points_density"):
        model.points_density = nn.Parameter(
            torch.ones(count, device=device, dtype=dtype), requires_grad=False
        )
        derived.append("points_density=ones")
    return derived


def _values_match(loaded: torch.Tensor, wanted: torch.Tensor) -> bool:
    """Bit-faithful equality that survives NaN.

    ``torch.equal`` reports NaN != NaN, and the archived ``points_acc_grad`` /
    ``points_acc_grad_norm`` buffers really do carry NaN (they accumulate raw gradients, and a
    pruned or never-hit point leaves 0/0 behind). Comparing them with ``torch.equal`` flags a
    bit-identical copy as a failed load, which would make every real checkpoint unloadable in
    strict mode. The NaNs are preserved verbatim in the tensor -- only this audit tolerates them.
    """
    if loaded.shape != wanted.shape or loaded.dtype != wanted.dtype:
        return False
    if loaded.is_floating_point() or loaded.is_complex():
        nan_loaded, nan_wanted = torch.isnan(loaded), torch.isnan(wanted)
        if not torch.equal(nan_loaded, nan_wanted):
            return False
        return torch.equal(loaded.masked_fill(nan_loaded, 0), wanted.masked_fill(nan_wanted, 0))
    return torch.equal(loaded, wanted)


def _apply_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    path: Path,
    strict: bool,
) -> dict[str, tuple[str, ...]]:
    """Migrate, resize and copy ``state`` into ``model``; return the audit trail.

    ``state`` is mutated: a migrated key is rewritten, and under ``strict=False`` a
    shape-mismatched key is dropped so ``load_state_dict`` cannot raise on it. The dropped names
    still reach the report, so a caller always learns which tensors kept their initialisation.
    """
    migrations = _migrate(state, model.state_dict())
    derived = _materialize_dynamic(model, state)

    current = model.state_dict()
    shape_mismatch = tuple(
        sorted(n for n in state if n in current and current[n].shape != state[n].shape)
    )
    if shape_mismatch:
        detail = ", ".join(
            f"{n}: checkpoint {tuple(state[n].shape)} vs model {tuple(current[n].shape)}"
            for n in shape_mismatch
        )
        if strict:
            raise RuntimeError(f"Checkpoint shape mismatch for {path}: {detail}")
        for name in shape_mismatch:
            state.pop(name)

    incompatible = model.load_state_dict(state, strict=False)
    missing = tuple(sorted(set(incompatible.missing_keys) - DERIVABLE_MISSING))
    unexpected = tuple(sorted(incompatible.unexpected_keys))

    loaded = model.state_dict()
    unequal = tuple(
        sorted(
            name
            for name, tensor in state.items()
            if name in loaded
            and not _values_match(loaded[name].detach().cpu(), tensor.detach().cpu())
        )
    )
    return {
        "migrations": tuple(migrations),
        "derived": tuple(derived),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "unequal": unequal,
    }


def load_checkpoint_strict(
    model: nn.Module,
    path: str | Path,
    *,
    expected_step: int | None = None,
    expected_sha256: str | None = None,
    require_step: bool = True,
) -> LoadReport:
    """Load an archived checkpoint, permitting only named migrations. Raises unless exact."""
    path = Path(path)
    actual = sha256_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected_sha256}")

    source, step, schema = read_checkpoint(
        path, expected_step=expected_step, require_step=require_step
    )
    state = {name: tensor.detach().clone() for name, tensor in source.items()}
    audit = _apply_state(model, state, path=path, strict=True)

    report = LoadReport(
        path=str(path.resolve()),
        sha256=actual,
        source_schema=schema,
        step=step,
        n_points=int(state["points"].shape[0]),
        **audit,
    )
    if not report.exact:
        raise RuntimeError(f"Non-exact checkpoint load: {report.as_dict()}")
    return report


def load_into_model(
    model: nn.Module,
    path: str | Path,
    *,
    expected_step: int | None = None,
    strict: bool = True,
    require_step: bool = True,
) -> LoadReport:
    """Load a checkpoint into a live model.

    The tensor work is :func:`read_checkpoint` plus :func:`_apply_state`, and the outcome is a
    :class:`LoadReport`. ``strict=True`` (the default) raises unless the load was exact.

    Per-point parameters are re-assigned rather than copied into: a checkpoint's point count is a
    product of the densification schedule and will not match a freshly initialised model's.

    Deliberately unsupported: ``training.exclude_keys``, which would let a caller skip any key
    whose name contained one of a list of substrings. It is ``[]`` in every shipped config, and an
    audited loader cannot silently leave a tensor at its initialisation. Also left to the caller is
    Setting ``model.pruned_points = True`` on the ``--load_path`` branch, which gates
    ``gt_sp_loss`` -- it is a training-schedule decision, not a property of the file.
    """
    path = Path(path)
    actual = sha256_file(path)
    source, step, schema = read_checkpoint(
        path, expected_step=expected_step, require_step=require_step
    )
    state = {name: tensor.detach().clone() for name, tensor in source.items()}
    audit = _apply_state(model, state, path=path, strict=strict)

    if "points_density=ones" in audit["derived"] and hasattr(model, "update_point_density"):
        model.update_point_density()

    report = LoadReport(
        path=str(path.resolve()),
        sha256=actual,
        source_schema=schema,
        step=step,
        n_points=int(state["points"].shape[0]),
        **audit,
    )
    if strict and not report.exact:
        raise RuntimeError(f"Non-exact checkpoint load: {report.as_dict()}")
    return report


def describe_checkpoint(path: str | Path) -> dict[str, Any]:
    """Inspect a checkpoint without a model. Used by the manifest and packaging tests."""
    try:
        state, step, schema = read_checkpoint(path)
    except ValueError:
        state, step, schema = read_checkpoint(path, expected_step=-1)
    shapes = {name: tuple(t.shape) for name, t in state.items()}
    return {
        "path": str(Path(path).resolve()),
        "schema": schema,
        "step": step,
        "n_keys": len(shapes),
        "n_points": int(state["points"].shape[0]) if "points" in state else None,
        "has_unet": any(n.startswith("unet.") for n in shapes),
        "empty_tensors": sorted(n for n, s in shapes.items() if 0 in s),
        "shapes": shapes,
    }

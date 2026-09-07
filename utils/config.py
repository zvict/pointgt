from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

from utils.runtime import ConfigNode, to_config_node

#: Repository root: ``utils/`` sits directly under it.
ROOT = Path(__file__).resolve().parents[1]

#: Config path prefixes interpreted relative to the repository root.
_PROJECT_PREFIXES = ("data/", "checkpoints/", "demo/", "experiments/")

#: ``2dgs/`` is resolved separately so a user can keep the 2D Gaussian Splatting output tree
#: outside this repository. See the depth-supervision section of the README.
_DEPTH_PREFIX = "2dgs/"
_DEPTH_ROOT_ENV = "POINTGT_2DGS_ROOT"


def depth_root() -> Path:
    """Where ``2dgs/...`` config paths resolve to."""
    return Path(os.environ.get(_DEPTH_ROOT_ENV, ROOT / "2dgs"))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input.

    Lists of named datasets merge *by name* rather than being replaced, matching the archived
    ``update_dict`` semantics that every ``test.datasets`` block relies on. Replacing them wholesale
    would drop inherited per-dataset fields and silently evaluate a different split.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            result[key] = deep_merge(current, value)
        elif key == "datasets" and isinstance(value, list) and isinstance(current, list):
            result[key] = _merge_named_list(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _merge_named_list(base: list[Any], override: list[Any]) -> list[Any]:
    merged = copy.deepcopy(base)
    template = copy.deepcopy(base[0]) if base else {}
    for entry in override:
        if not isinstance(entry, dict) or "name" not in entry:
            merged.append(copy.deepcopy(entry))
            continue
        for existing in merged:
            if isinstance(existing, dict) and existing.get("name") == entry["name"]:
                existing.update(copy.deepcopy(entry))
                break
        else:
            merged.append(deep_merge(template, entry) if template else copy.deepcopy(entry))
    return merged


def set_dotted(config: dict[str, Any], expression: str) -> None:
    """Apply a ``models.attn.k_type=13`` style override in place."""
    if "=" not in expression:
        raise ValueError(f"Override must be KEY=VALUE, got {expression!r}")
    dotted, raw = expression.split("=", 1)
    keys = dotted.split(".")
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
        if not isinstance(target, dict):
            raise ValueError(f"Cannot set {dotted!r}: {key!r} is not a mapping")
    target[keys[-1]] = yaml.safe_load(raw)


def load_defaults(*, uv: bool = False) -> dict[str, Any]:
    """The archived defaults. The UV stage merges against a different file, not a superset."""
    name = "default_uv.yml" if uv else "default.yml"
    return _load_yaml(ROOT / "configs" / name)


def resolve_paths(value: Any, project_root: Path | None = None) -> Any:
    """Rewrite release-relative config paths into absolute paths."""
    root = Path(project_root) if project_root is not None else ROOT
    if isinstance(value, dict):
        return {k: resolve_paths(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_paths(v, root) for v in value]
    if not isinstance(value, str):
        return value
    if value.startswith(_DEPTH_PREFIX):
        return str((depth_root() / value.removeprefix(_DEPTH_PREFIX)).expanduser())
    if value.startswith(_PROJECT_PREFIXES):
        return str((root / value).expanduser())
    return value


def config_fingerprint(config: dict[str, Any]) -> str:
    """Hash the portable config, excluding location-only fields.

    Where a run reads from or writes to does not change what it computes, and neither does what it
    is called, so both are omitted; two runs with the same fingerprint are the same experiment.
    ``obj_name``, ``save_name``, ``modifier`` and the whole ``wandb`` block are naming metadata that
    no code in this release reads.
    """
    snapshot = copy.deepcopy(config)
    for key in ("load_path", "save_dir", "index", "obj_name", "save_name", "wandb", "modifier"):
        snapshot.pop(key, None)
    snapshot.pop("release", None)
    if isinstance(snapshot.get("dataset"), dict):
        snapshot["dataset"].pop("gt_depth_dir", None)
    canonical = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class ResolvedConfig:
    values: ConfigNode
    source: Path
    config_sha256: str
    fingerprint: str

    @property
    def scene(self) -> str:
        return self.source.stem

    @property
    def benchmark(self) -> str:
        return self.source.parent.name


def load_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    overrides: list[str] | None = None,
    uv: bool | None = None,
) -> ResolvedConfig:
    """Merge a scene config over the archived defaults and resolve its paths."""
    path = Path(path)
    raw_text = path.read_bytes()
    scene = _load_yaml(path)

    is_uv = path.stem.endswith("_uv") if uv is None else bool(uv)
    if "nuvo" in scene and not is_uv:
        raise ValueError(
            f"{path} defines a 'nuvo:' block but is not being merged against configs/default_uv.yml. "
            "Stage-b configs are recognised by a '_uv' filename suffix; rename the file, or pass "
            "uv=True."
        )
    merged = deep_merge(load_defaults(uv=is_uv), scene)

    for expression in overrides or []:
        set_dotted(merged, expression)

    fingerprint = config_fingerprint(merged)
    resolved = resolve_paths(merged, project_root)

    return ResolvedConfig(
        values=to_config_node(resolved),
        source=path.resolve(),
        config_sha256=hashlib.sha256(raw_text).hexdigest(),
        fingerprint=fingerprint,
    )

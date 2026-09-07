from __future__ import annotations

import os
import random
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any, IO, TextIO

import numpy as np
import torch

from utils.config import ROOT, ResolvedConfig

#: Directory names pruned from the code snapshot walk. ``.git`` is added because the release tree
#: may sit inside a checkout and a git object store holds no ``.py`` files worth archiving.
_SNAPSHOT_SKIP_DIRS = frozenset({"__pycache__", "experiments", "data", ".git"})


def setup_seed(seed: int) -> None:
    """Seed torch (CPU + all CUDA devices), numpy and ``random``.

    Order matters only in that all four must be seeded before anything draws.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_torch(config: Any, device: torch.device | None = None) -> torch.device:
    """Apply the global torch settings the released checkpoints were trained under.

    MUST be called before the model is constructed -- see the module docstring. Returns the device
    so the call site naturally reads ``device = setup_torch(cfg)`` on the line above ``PAPR(...)``.

    ``torch.set_float32_matmul_precision('high')`` is numerics-affecting and is applied only on
    CUDA; on CPU the setting is inert, so it is left unset there. This is *not* wrapped in a
    try/except: silently continuing after a failed TF32 enable would train a model at different
    precision than every released checkpoint while looking like a harmless warning. Here the
    failure stops the run.

    The dynamo settings below are applied unconditionally. ``training.use_compile`` is an
    archived key that this release does not read; there is no compile path in ``train.py``. (``false`` in
    ``configs/default.yml`` and in every shipped config). They are kept unconditionally so that a
    user who enables compilation gets consistent graph-break and cache-size behaviour rather than
    torch's defaults.
    """
    values = _values(config)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    training = values.get("training", {}) if hasattr(values, "get") else {}
    try:
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.accumulated_cache_size_limit = int(
            _get(training, "compile_cache_size_limit", 512)
        )
        torch._dynamo.config.cache_size_limit = int(
            _get(training, "compile_per_function_cache_limit", 64)
        )
    except Exception as error:  # noqa: BLE001 - inert settings; never worth failing a run over
        print(f"Warning: could not configure torch._dynamo ({error})")
    return device


class Logger:
    """Tee a text stream to a log file, keeping the original stream live.

    Installed as ``sys.stdout = Logger(path, sys.stdout)``. The file is opened in append mode so a
    resumed run extends its log rather than truncating the history it is resuming from.

    The wrapper forwards the ``isatty``/``fileno``/``readable``/... surface because tqdm and
    ``subprocess`` inspect it; a plain object without ``fileno`` breaks any child process that
    inherits stdout.

    Also usable as a context manager, which restores the original stream and records a traceback
    into the log on the way out.
    """

    def __init__(self, filename: str | Path, stream: TextIO = sys.stdout) -> None:
        self.file: IO[str] = open(filename, "a", encoding="utf-8")
        self.stream = stream
        self._saved: TextIO | None = None

    def __enter__(self) -> "Logger":
        self._saved = sys.stdout
        sys.stdout = self  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc_value, tb) -> bool:
        if self._saved is not None:
            sys.stdout = self._saved
            self._saved = None
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()
        return False

    def write(self, data: str) -> int:
        self.file.write(data)
        self.stream.write(data)
        return len(data)

    def flush(self) -> None:
        self.file.flush()
        self.stream.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()

    def readable(self) -> bool:
        return getattr(self.stream, "readable", lambda: False)()

    def writable(self) -> bool:
        return getattr(self.stream, "writable", lambda: True)()

    def seekable(self) -> bool:
        return getattr(self.stream, "seekable", lambda: False)()

    def read(self, size: int = -1) -> str:
        return self.stream.read(size)

    def readline(self, size: int = -1) -> str:
        return self.stream.readline(size)

    def readlines(self, hint: int = -1) -> list[str]:
        return self.stream.readlines(hint)

    def tell(self) -> int:
        return getattr(self.stream, "tell", lambda: 0)()

    def seek(self, offset: int, whence: int = 0) -> int:
        return getattr(self.stream, "seek", lambda _o, _w: 0)(offset, whence)

    def truncate(self, size: int | None = None) -> int:
        return getattr(self.stream, "truncate", lambda _s: 0)(size)

    def close(self) -> None:
        """Close the log file. The wrapped stream is left open; it is usually the real stdout."""
        self.flush()
        self.file.close()


def make_run_dir(config: Any) -> Path:
    """Create and return ``<save_dir>/<index>``, the run's output directory.

    Everything a run produces lands here: checkpoints, renders, ``train.log``, ``code.zip`` and the
    archived config. A relative ``save_dir`` (the shipped configs all use ``./experiments/...``) is
    resolved against the release root rather than the process CWD, so a run started from a
    subdirectory writes to the same place as one started from the root.
    """
    values = _values(config)
    save_dir = Path(str(values["save_dir"]))
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    run_dir = (save_dir / str(values["index"])).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_code(dest: str | Path, src_dir: str | Path = ROOT) -> Path:
    """Zip every ``.py`` file under ``src_dir`` into ``dest``, archiving the code beside the run.

    This lets a checkpoint always be paired with the source that produced it (called with
    ``src_dir="."`` from the release root). Entries are stored under paths relative to ``src_dir``.

    The file list is sorted, so re-running on an unchanged tree produces an identical archive
    instead of one that depends on directory iteration order.
    """
    src_dir = Path(src_dir).resolve()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    python_files: list[Path] = []
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SNAPSHOT_SKIP_DIRS)
        for name in files:
            if name.endswith(".py"):
                python_files.append(Path(root) / name)

    with zipfile.ZipFile(dest, "w") as archive:
        for path in sorted(python_files):
            archive.write(path, path.relative_to(src_dir).as_posix())
    return dest


def archive_config(config: ResolvedConfig, run_dir: str | Path) -> Path:
    """Copy the scene config into the run directory, next to ``code.zip``.

    The source bytes are written verbatim -- so the copy still hashes to ``config_sha256`` -- with
    the resolved identity appended as trailing YAML comments, which is what actually distinguishes
    two runs whose config files differ only in ``index``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / config.source.name
    provenance = (
        f"\n# --- resolved by utils.session.archive_config ---\n"
        f"# source: {config.source}\n"
        f"# config_sha256: {config.config_sha256}\n"
        f"# fingerprint: {config.fingerprint}\n"
    )
    dest.write_bytes(config.source.read_bytes() + provenance.encode("utf-8"))
    return dest


def _values(config: Any) -> Any:
    """Accept either a :class:`ResolvedConfig` or the bare ``ConfigNode`` it wraps."""
    return config.values if isinstance(config, ResolvedConfig) else config


def _get(node: Any, key: str, default: Any) -> Any:
    """``node.get(key, default)`` for mappings, ``getattr`` for anything else."""
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)

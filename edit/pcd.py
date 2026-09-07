from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from utils.ply import read_points

__all__ = [
    "POINT_CLOUD_SUFFIXES",
    "SEQUENCE_DIR_NAMES",
    "find_sequence_dir",
    "list_point_clouds",
    "load_point_cloud",
    "load_deformed_points",
]

#: Extensions this module discovers point-cloud files by.
#: ``.pcd`` is kept in the discoverable set so a
#: directory written by a point-cloud tool is still enumerated, but reading one needs open3d --
#: see :func:`load_point_cloud`.
POINT_CLOUD_SUFFIXES: tuple[str, ...] = (".ply", ".xyz", ".pcd")

#: Sequence directory names tried, in order, when the caller does not name one. ``rbf_pcds_fixcp``
#: (a re-run with the control points pinned) is preferred over ``rbf_pcds``; a run that has both
#: wants the newer one.
SEQUENCE_DIR_NAMES: tuple[str, ...] = ("rbf_pcds_fixcp", "rbf_pcds")


def find_sequence_dir(base_dir: str | Path) -> Path | None:
    """First of :data:`SEQUENCE_DIR_NAMES` that exists under ``base_dir``, or ``None``."""
    base = Path(base_dir).expanduser()
    for name in SEQUENCE_DIR_NAMES:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return None


def list_point_clouds(directory: str | Path) -> list[Path]:
    """Every point-cloud file in ``directory``, ordered the way the sequence is numbered.

    Sorted by **lower-cased basename**. It is not the same as sorting by
    path: a case-sensitive sort would interleave ``Edit_0002.ply`` before ``edit_0001.ply`` and
    silently permute the animation. Zero-padded names (``edit_0001.ply``) then sort numerically.

    Returns an empty list for a path that is not a directory -- the caller reports the empty
    sequence with the directory it looked in, which is a better error than a bare
    ``NotADirectoryError`` from here.
    """
    resolved = Path(directory).expanduser().resolve()
    if not resolved.is_dir():
        return []
    paths = [p for p in resolved.iterdir() if p.suffix.lower() in POINT_CLOUD_SUFFIXES]
    paths.sort(key=lambda path: path.name.lower())
    return paths


def _read_pcd_via_open3d(path: Path) -> np.ndarray:
    """``.pcd`` fallback. Only this extension needs open3d; ``.ply``/``.xyz`` go through numpy."""
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - depends on the environment, not the input
        raise ImportError(
            f"Reading {path.name} needs open3d, which this release does not require for anything "
            "else. Install it (`pip install open3d`) or convert the sequence to .ply/.xyz."
        ) from exc
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def load_point_cloud(
    path: str | Path,
    *,
    coord_scale: float = 1.0,
    device: str | torch.device = "cuda",
    scale_up: bool = True,
) -> Tensor:
    """Read one point cloud into the renderer's scaled frame.

    Args:
        path: A ``.ply``, ``.xyz`` or ``.pcd`` file.
        coord_scale: The model's ``coord_scale``.
        device: Where the returned tensor lives.
        scale_up: Multiply by ``coord_scale``. False only when the caller already holds points in
            the renderer's frame (model points, for instance) and is round-tripping them.

    Returns:
        ``(N, 3)`` float32.
    """
    path = Path(path).expanduser()
    if path.suffix.lower() == ".pcd":
        raw = _read_pcd_via_open3d(path)
    else:
        raw = read_points(path)
    points = torch.from_numpy(np.ascontiguousarray(raw).astype(np.float32)).to(device)
    if scale_up:
        points = points * coord_scale
    return points


def load_deformed_points(
    path: str | Path,
    *,
    expected_count: int,
    coord_scale: float = 1.0,
    device: str | torch.device = "cuda",
) -> Tensor:
    """Load one deformed frame and enforce the correspondence contract.

    The renderer substitutes these positions for the checkpoint's while leaving ``pc_feats``,
    ``points_influ_scores``, ``points_scaler`` and ``bkg_points_mask`` on the checkpoint's rows,
    and the canonical transfer indexes the checkpoint's snapshot with indices selected against the
    *deformed* positions. All of that is only meaningful if row ``i`` means the same point in both
    clouds, and a count mismatch is the one violation that is detectable. Left unchecked it does
    not raise anywhere downstream: it either silently pairs the wrong features with the wrong
    points, or trips an opaque shape error deep in the attention.
    """
    points = load_point_cloud(path, coord_scale=coord_scale, device=device, scale_up=True)
    if points.shape[0] != expected_count:
        raise ValueError(
            f"{Path(path).name} has {points.shape[0]} points but the checkpoint's cloud has "
            f"{expected_count}. A deformed frame must be the checkpoint's own cloud moved, point "
            "for point: the per-point features, influence scores and scalers are not reloaded "
            "per frame, and the canonical transfer indexes the checkpoint's snapshot with indices "
            "chosen against these positions. Re-export the sequence from the cloud this "
            "checkpoint holds."
        )
    return points

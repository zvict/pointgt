from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np

#: PLY scalar type tokens -> numpy type codes. Both the classic names and the explicit-width
#: aliases appear in the wild; open3d accepted both, so this does too.
_PLY_TO_NUMPY: dict[str, str] = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}

_FORMATS = ("ascii", "binary_little_endian", "binary_big_endian")


@dataclass(frozen=True)
class _Property:
    """One PLY property. ``count_type`` is non-``None`` only for list properties."""

    name: str
    type: str
    count_type: str | None = None

    @property
    def is_list(self) -> bool:
        return self.count_type is not None


@dataclass(frozen=True)
class _Element:
    name: str
    count: int
    properties: tuple[_Property, ...]

    @property
    def has_list(self) -> bool:
        return any(p.is_list for p in self.properties)

    def numpy_dtype(self) -> np.dtype:
        """Structured dtype for one fixed-size instance. Little-endian; callers byte-swap."""
        return np.dtype([(p.name, "<" + _PLY_TO_NUMPY[p.type]) for p in self.properties])


def _parse_header(handle: BinaryIO, path: Path) -> tuple[str, list[_Element]]:
    """Consume the header, leaving the handle positioned at the first data byte."""
    magic = handle.readline().strip()
    if magic != b"ply":
        raise ValueError(f"{path}: not a PLY file (first line is {magic!r})")

    fmt: str | None = None
    elements: list[_Element] = []
    pending: list[_Property] = []
    names: list[str] = []
    counts: list[int] = []

    def _flush() -> None:
        if names:
            elements.append(_Element(names[-1], counts[-1], tuple(pending)))

    while True:
        raw = handle.readline()
        if not raw:
            raise ValueError(f"{path}: PLY header ended without 'end_header'")
        tokens = raw.decode("ascii", errors="replace").strip().split()
        if not tokens or tokens[0] == "comment" or tokens[0] == "obj_info":
            continue
        keyword = tokens[0]
        if keyword == "format":
            fmt = tokens[1]
            if fmt not in _FORMATS:
                raise ValueError(f"{path}: unknown PLY format {fmt!r}")
        elif keyword == "element":
            _flush()
            pending = []
            names.append(tokens[1])
            counts.append(int(tokens[2]))
        elif keyword == "property":
            if not names:
                raise ValueError(f"{path}: 'property' before any 'element'")
            if tokens[1] == "list":
                pending.append(_Property(tokens[4], tokens[3], count_type=tokens[2]))
            else:
                if tokens[1] not in _PLY_TO_NUMPY:
                    raise ValueError(f"{path}: unsupported PLY property type {tokens[1]!r}")
                pending.append(_Property(tokens[2], tokens[1]))
        elif keyword == "end_header":
            _flush()
            break

    if fmt is None:
        raise ValueError(f"{path}: PLY header has no 'format' line")
    return fmt, elements


def _read_binary_vertices(
    handle: BinaryIO, path: Path, fmt: str, elements: list[_Element]
) -> dict[str, np.ndarray]:
    little = fmt == "binary_little_endian"
    for element in elements:
        if element.name != "vertex":
            if element.has_list:
                raise NotImplementedError(
                    f"{path}: element {element.name!r} with list properties precedes 'vertex'"
                )
            handle.seek(element.count * element.numpy_dtype().itemsize, io.SEEK_CUR)
            continue
        if element.has_list:
            raise NotImplementedError(f"{path}: 'vertex' element has list properties")
        dtype = element.numpy_dtype()
        if not little:
            dtype = dtype.newbyteorder(">")
        data = np.fromfile(handle, dtype=dtype, count=element.count)
        if data.shape[0] != element.count:
            raise ValueError(
                f"{path}: truncated vertex block ({data.shape[0]} of {element.count} vertices)"
            )
        return {name: np.ascontiguousarray(data[name]) for name in dtype.names}
    raise ValueError(f"{path}: PLY has no 'vertex' element")


def _read_ascii_vertices(
    handle: BinaryIO, path: Path, elements: list[_Element]
) -> dict[str, np.ndarray]:
    text = handle.read().decode("ascii", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    cursor = 0
    for element in elements:
        block = lines[cursor : cursor + element.count]
        cursor += element.count
        if element.name != "vertex":
            continue
        if len(block) != element.count:
            raise ValueError(
                f"{path}: truncated vertex block ({len(block)} of {element.count} vertices)"
            )
        if element.has_list:
            raise NotImplementedError(f"{path}: 'vertex' element has list properties")
        if element.count == 0:
            return {p.name: np.empty(0, dtype=_PLY_TO_NUMPY[p.type]) for p in element.properties}
        table = np.loadtxt(io.StringIO("\n".join(block)), dtype=np.float64, ndmin=2)
        if table.shape[1] != len(element.properties):
            raise ValueError(
                f"{path}: vertex row has {table.shape[1]} fields, header declares "
                f"{len(element.properties)}"
            )
        return {
            prop.name: table[:, i].astype(_PLY_TO_NUMPY[prop.type])
            for i, prop in enumerate(element.properties)
        }
    raise ValueError(f"{path}: PLY has no 'vertex' element")


def read_ply_vertices(path: str | Path) -> dict[str, np.ndarray]:
    """Read every vertex property of a PLY file into a dict of 1-D arrays.

    Supports ``ascii`` and ``binary_little_endian`` (the two formats the released assets use) plus
    ``binary_big_endian``, which costs one byte-swap. Elements other than ``vertex`` are skipped.
    """
    path = Path(path)
    with path.open("rb") as handle:
        fmt, elements = _parse_header(handle, path)
        if fmt == "ascii":
            return _read_ascii_vertices(handle, path, elements)
        return _read_binary_vertices(handle, path, fmt, elements)


def read_ply_points(path: str | Path) -> np.ndarray:
    """Read vertex positions as ``(N, 3)`` ``float64``.

    ``float64`` regardless of the file's storage type, matching what
    ``open3d.io.read_point_cloud(...).points`` returns.
    """
    fields = read_ply_vertices(path)
    missing = [axis for axis in "xyz" if axis not in fields]
    if missing:
        raise ValueError(f"{path}: PLY vertex element is missing {missing}")
    return np.stack(
        [fields["x"].astype(np.float64), fields["y"].astype(np.float64),
         fields["z"].astype(np.float64)],
        axis=-1,
    )


def read_ply_colors(path: str | Path) -> np.ndarray | None:
    """Read vertex colours as ``(N, 3)`` ``float64`` in ``[0, 1]``, or ``None`` if absent."""
    fields = read_ply_vertices(path)
    if not all(name in fields for name in ("red", "green", "blue")):
        return None
    stacked = np.stack([fields["red"], fields["green"], fields["blue"]], axis=-1)
    if stacked.dtype == np.uint8:
        return stacked.astype(np.float64) / 255.0
    return stacked.astype(np.float64)


def write_ply_points(
    path: str | Path,
    points: Any,
    colors: Any | None = None,
    normals: Any | None = None,
    *,
    fmt: Literal["binary", "ascii"] = "binary",
) -> Path:
    """Write ``(N, 3)`` positions, with optional normals and colours, to a PLY file.

    The layout when ``fmt="binary"`` is: ``double`` x/y/z, then ``double``
    nx/ny/nz if given, then ``uchar`` red/green/blue if given. Normals are written as nx/ny/nz so
    viewers show per-point orientation. Colours are clipped to ``[0, 1]`` and scaled to 0-255 --
    a colour array already in 0-255 will therefore saturate, which is intentional, not an oversight.

    ``fmt="ascii"`` emits the same properties in the same order using ``%.17g`` for the doubles,
    which round-trips ``float64`` exactly.
    """
    if fmt not in ("binary", "ascii"):
        raise ValueError(f"fmt must be 'binary' or 'ascii', got {fmt!r}")
    path = Path(path)
    pts = np.asarray(_to_numpy(points), dtype=np.float64).reshape(-1, 3)

    fields: list[tuple[str, str]] = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
    props = ["property double x", "property double y", "property double z"]
    if normals is not None:
        fields += [("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8")]
        props += ["property double nx", "property double ny", "property double nz"]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        props += ["property uchar red", "property uchar green", "property uchar blue"]

    data = np.empty(pts.shape[0], dtype=fields)
    data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if normals is not None:
        nrm = np.asarray(_to_numpy(normals), dtype=np.float64).reshape(-1, 3)
        _check_rows(nrm, pts, "normals", path)
        data["nx"], data["ny"], data["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    if colors is not None:
        rgb = np.asarray(_to_numpy(colors)).reshape(-1, 3)
        _check_rows(rgb, pts, "colors", path)
        rgb = (np.clip(rgb.astype(np.float64), 0.0, 1.0) * 255).astype(np.uint8)
        data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    ply_format = "binary_little_endian 1.0" if fmt == "binary" else "ascii 1.0"
    header = "\n".join(
        ["ply", f"format {ply_format}", f"element vertex {pts.shape[0]}", *props, "end_header", ""]
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        if fmt == "binary":
            data.tofile(handle)
        else:
            handle.write(_ascii_rows(data).encode("ascii"))
    return path


def _ascii_rows(data: np.ndarray) -> str:
    """Format a structured vertex array as ascii rows, one vertex per line."""
    lines = []
    for row in data:
        cells = []
        for name in data.dtype.names:
            value = row[name]
            cells.append(str(int(value)) if data.dtype[name].kind == "u" else f"{value:.17g}")
        lines.append(" ".join(cells))
    return "\n".join(lines) + ("\n" if lines else "")


def _check_rows(array: np.ndarray, points: np.ndarray, what: str, path: Path) -> None:
    if array.shape[0] != points.shape[0]:
        raise ValueError(
            f"{path}: {what} has {array.shape[0]} rows but there are {points.shape[0]} points"
        )


def _to_numpy(value: Any) -> Any:
    """Accept torch tensors without importing torch.

    ``utils.ply`` is imported by tooling that has no reason to pay for a torch import, so the
    tensor case is duck-typed rather than type-checked.
    """
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return value


def read_points(path: str | Path) -> np.ndarray:
    """Load ``(N, 3)`` points from ``.ply``, ``.xyz``/``.txt``, ``.npy`` or ``.npz``.

    Handles everything except ``.pth``/``.pt``; that branch lives in
    :func:`models.init.load_points` where torch is already a dependency.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xyz", ".txt"):
        return np.loadtxt(path, delimiter=" ", dtype=float).reshape(-1, 3)
    if suffix == ".ply":
        return read_ply_points(path)
    if suffix == ".npy":
        return np.load(path).reshape(-1, 3)
    if suffix == ".npz":
        with np.load(path) as data:
            return data["points"].reshape(-1, 3)
    raise ValueError(f"Unsupported file format: {path.suffix.lstrip('.')}")


def save_points(path: str | Path, points: Any) -> Path:
    """Write points to ``.xyz`` (whitespace-delimited text) or ``.ply`` (binary little-endian).

    Uses ``np.savetxt``'s default ``%.18e`` formatting for ``.xyz``.
    """
    path = Path(path)
    array = np.asarray(_to_numpy(points)).reshape(-1, 3)
    suffix = path.suffix.lower()
    if suffix == ".xyz":
        np.savetxt(path, array, delimiter=" ")
        return path
    if suffix == ".ply":
        return write_ply_points(path, array)
    raise ValueError(f"Unsupported file format: {path.suffix.lstrip('.')}")

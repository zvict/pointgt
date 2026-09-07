from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

__all__ = [
    "TEXTURE_GRID_COLUMNS",
    "grid_layout",
    "validate_grid_image",
    "pack_texture_map_to_grid",
    "unpack_grid_to_texture_maps",
    "create_edit_mask_from_texture_maps",
    "tile_image_to_charts",
]

#: Widest a packed grid gets. The archived editor backend and the archived figure renderer both
#: hard-coded four columns, so every grid the paper's figures were painted on is four-wide (or, for
#: an atlas of more than four charts, four-wide over several rows). Changing this changes which cell
#: holds which chart, i.e. it invalidates every previously exported PNG.
TEXTURE_GRID_COLUMNS = 4


def grid_layout(num_charts: int, columns: int | None = None) -> tuple[int, int]:
    """The ``(rows, columns)`` a ``num_charts`` atlas packs into.

    The single source of the layout. ``columns`` defaults to ``min(num_charts,
    TEXTURE_GRID_COLUMNS)``, which is the archived four-wide convention for every atlas of four or
    more charts -- including the shipped four-chart dress, which lands on one row of four -- and
    avoids padding a smaller atlas out with blank cells nothing would ever read.

    Args:
        num_charts: Charts in the atlas (``nuvo.model.num_charts``).
        columns: Override, for reading a grid that was packed some other way. Must be positive.

    Returns:
        ``(rows, columns)``. ``rows * columns >= num_charts``; any trailing cells are unused.
    """
    if num_charts < 1:
        raise ValueError(f"num_charts must be >= 1, got {num_charts}")
    if columns is None:
        columns = min(num_charts, TEXTURE_GRID_COLUMNS)
    columns = int(columns)
    if columns < 1:
        raise ValueError(f"columns must be >= 1, got {columns}")
    rows = (num_charts + columns - 1) // columns
    return rows, columns


def validate_grid_image(grid_shape: tuple[int, ...], num_charts: int,
                        columns: int | None = None) -> tuple[int, int]:
    """Check a packed grid's pixel shape against the layout, or say why it does not fit.

    Reading a grid with the wrong layout fails *silently*: the unpack succeeds, the render succeeds,
    and the user's paint lands on the wrong part of the surface. There is enough structure to catch
    it -- the cells tile the image exactly and charts are square (``texture_map_res_per_chart`` is
    one number) -- so an image that cannot be the expected layout is rejected here instead.

    Args:
        grid_shape: The image's ``.shape``; only the first two entries are read.
        num_charts: Charts the grid is expected to hold.
        columns: Override passed through to :func:`grid_layout`.

    Returns:
        ``(rows, columns)``, the validated layout.

    Raises:
        ValueError: The image is not an exact ``rows x columns`` tiling of square cells.
    """
    height, width = int(grid_shape[0]), int(grid_shape[1])
    rows, cols = grid_layout(num_charts, columns)
    expected = f"{num_charts} charts pack as {rows} row(s) x {cols} column(s) of square charts"
    if height % rows or width % cols:
        raise ValueError(
            f"texture grid is {width}x{height} px, which {rows} row(s) x {cols} column(s) do not "
            f"divide evenly ({expected}). Re-export it from train_uv.py, or pass "
            "--texture_grid_columns to say how it was packed."
        )
    cell_h, cell_w = height // rows, width // cols
    if cell_h != cell_w:
        raise ValueError(
            f"texture grid is {width}x{height} px, giving {cell_w}x{cell_h} px cells, but "
            f"{expected}. Either it was packed with a different column count -- say which with "
            "--texture_grid_columns -- or it is not a packed grid at all (a figure of the charts, "
            "say, rather than the charts themselves). Re-export it from train_uv.py, or drop "
            "--texture_is_grid to tile the image across every chart instead."
        )
    return rows, cols


def pack_texture_map_to_grid(texture_maps: np.ndarray, columns: int | None = None) -> np.ndarray:
    """Tile a chart stack into one editable image.

    Args:
        texture_maps: ``(num_charts, H, W, C)`` in ``[0, 1]``.
        columns: Override for the column count; defaults to :func:`grid_layout` of the stack's own
            chart count, which is what :func:`unpack_grid_to_texture_maps` assumes.

    Returns:
        ``(grid_H, grid_W, C)`` ``uint8``. Cells past the last chart stay black.
    """
    num_charts, chart_h, chart_w, channels = texture_maps.shape
    rows, cols = grid_layout(num_charts, columns)

    grid = np.zeros((rows * chart_h, cols * chart_w, channels), dtype=np.float32)
    for i in range(num_charts):
        row, col = i // cols, i % cols
        grid[row * chart_h:(row + 1) * chart_h, col * chart_w:(col + 1) * chart_w] = texture_maps[i]

    return (np.clip(grid, 0, 1) * 255).astype(np.uint8)


def unpack_grid_to_texture_maps(grid_image: np.ndarray, num_charts: int,
                                columns: int | None = None) -> np.ndarray:
    """Cut a packed grid back into a chart stack.

    The cell size is recovered by integer division, so a grid whose dimensions are not exact
    multiples of the layout loses its last row/column of pixels rather than raising -- which is what
    happens when an editor exports at a rounded resolution. Call
    :func:`validate_grid_image` first when the image did not come from
    :func:`pack_texture_map_to_grid`.

    Args:
        grid_image: ``(grid_H, grid_W, C)``, ``uint8`` in ``[0, 255]``.
        num_charts: How many cells to read, in row-major order.
        columns: The same override :func:`pack_texture_map_to_grid` was given, if any.

    Returns:
        ``(num_charts, H, W, C)`` ``float32`` in ``[0, 1]``.
    """
    grid_h, grid_w, _ = grid_image.shape
    rows, cols = grid_layout(num_charts, columns)

    chart_h = grid_h // rows
    chart_w = grid_w // cols

    charts = []
    for i in range(num_charts):
        row, col = i // cols, i % cols
        charts.append(grid_image[row * chart_h:(row + 1) * chart_h,
                                 col * chart_w:(col + 1) * chart_w])

    return np.stack(charts, axis=0).astype(np.float32) / 255.0


def create_edit_mask_from_texture_maps(original_texture_map: Tensor, edited_texture_map: Tensor,
                                       threshold: float = 0.01) -> Tensor:
    """Which texels a person changed, as a 0/1 mask.

    The editing render composites the edited texture only where this is 1, so an edit stays local
    and the untouched parts of the atlas keep rendering the learned appearance. The test is the L2
    distance per texel across channels, which makes a hue change and a brightness change comparable
    and lets one threshold serve both.

    Args:
        original_texture_map: ``(num_charts, H, W, C)`` in ``[0, 1]``.
        edited_texture_map: Same shape, after editing.
        threshold: L2 distance above which a texel counts as edited. PNG round-tripping quantises to
            1/255 ~= 0.004, so the 0.01 default is above the noise floor of an unedited export.

    Returns:
        ``(num_charts, H, W, 1)`` float mask, 1 where edited.
    """
    diff = torch.norm(original_texture_map - edited_texture_map, dim=-1, keepdim=True)
    return (diff > threshold).float()


def tile_image_to_charts(image: np.ndarray | Image.Image | str | Path, num_charts: int,
                         chart_res: int, columns: int | None = None) -> np.ndarray:
    """Repeat one image across every chart, as a packed grid ready to load.

    It exists so the editing pipeline can be driven with any picture without the interactive
    editor: resize, copy into all ``num_charts`` cells, pack.

    **It does not reproduce the paper's dress figure.** Each of that figure's four charts carries a
    *different* affine transform of the source image -- placed, rotated and scaled per chart in the
    editor so the pattern lines up across the seams of the atlas. Tiling the same image into every
    chart puts the pattern on the object, but the charts disagree at their boundaries, so the seams
    show. Use it to sanity-check the round trip and as a starting point to edit from.

    Args:
        image: RGB source. An array (``uint8`` in ``[0, 255]`` or float in ``[0, 1]``), a PIL image,
            or a path to one. An alpha channel is dropped.
        num_charts: Charts in the atlas (``nuvo.model.num_charts``).
        chart_res: Side length of one chart, i.e. ``texture_map_res // sqrt(num_charts)``, which is
            what :class:`uv.nuvo.Nuvo` stores as ``texture_map_res_per_chart``.
        columns: Override for the column count; the default is the layout everything else uses.

    Returns:
        ``(grid_H, grid_W, 3)`` ``uint8``, the same layout
        :func:`unpack_grid_to_texture_maps` reads.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
        image = Image.fromarray(array)

    resized = np.asarray(image.convert("RGB").resize((chart_res, chart_res), Image.LANCZOS))
    charts = np.repeat(resized[None, ...], num_charts, axis=0).astype(np.float32) / 255.0
    return pack_texture_map_to_grid(charts, columns=columns)

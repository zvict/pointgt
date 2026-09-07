from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay
from torch import Tensor

from utils.ply import read_ply_points

__all__ = ["load_bounding_box", "points_inside_bounding_box"]


def load_bounding_box(mesh_path: str | Path, coord_scale: float = 1.0) -> Delaunay:
    """Read a box mesh's vertices and return the hull that tests containment.

    Args:
        mesh_path: A ``.ply`` whose vertices span the box (8 for a cube).
        coord_scale: Multiplied into the vertices before the hull is built. The surface points this
            filters are already divided by ``coord_scale`` (see
            :func:`models.surface.get_surface_points`), so the shipped call site passes ``1.0`` --
            the box is authored in scene units. Pass the model's ``coord_scale`` only if the box was
            drawn in the renderer's scaled frame.

    Returns:
        A :class:`scipy.spatial.Delaunay` triangulation of the vertices, for
        :func:`points_inside_bounding_box`.
    """
    vertices = read_ply_points(mesh_path) * coord_scale
    print(f"Loaded bounding box from {mesh_path} with {len(vertices)} vertices")
    print(f"  Bounding box extents (after scaling by coord_scale={coord_scale}):")
    print(f"    min: {vertices.min(axis=0)}")
    print(f"    max: {vertices.max(axis=0)}")
    return Delaunay(vertices)


def points_inside_bounding_box(points: Tensor, hull: Delaunay) -> Tensor:
    """Boolean mask of the points inside ``hull``.

    Runs on the CPU through scipy: ``find_simplex`` is a walk over the triangulation with no torch
    equivalent, and it is called a few dozen times per run (once per view), not per step.

    Args:
        points: ``(..., 3)``. Any leading shape; the mask comes back with that shape.
        hull: From :func:`load_bounding_box`.

    Returns:
        A boolean tensor shaped like ``points.shape[:-1]``, on ``points``' device.
    """
    original_shape = points.shape[:-1]
    points_np = points.detach().cpu().numpy().reshape(-1, 3)
    inside = hull.find_simplex(points_np) >= 0
    return torch.from_numpy(np.asarray(inside)).to(points.device).reshape(original_shape)

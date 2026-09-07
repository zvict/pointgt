from __future__ import annotations

import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from utils.runtime import ConfigNode

__all__ = ["load_blender_data", "load_meta_data"]


def load_blender_data(
    basedir: str | os.PathLike[str],
    split: str = "train",
    factor: int = 1,
    read_offline: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[float], list[str]]:
    """Read ``transforms_{split}.json`` and the images it references.

    Args:
        basedir: Scene directory holding the transforms JSON.
        split: ``train`` / ``test`` / ``val``; selects the transforms file.
        factor: Integer downsample applied to every image (``1`` disables it).
        read_offline: Load every image into memory. When ``False`` only the *first* image is read
            -- just enough to learn ``H``/``W`` -- while ``image_paths`` still lists all frames.
            That leaves ``images`` and ``image_paths`` different lengths, which every consumer
            here indexes as if they matched; it is a latent bug, kept because ``read_offline`` is
            ``true`` in every shipped config and flipping the behaviour would change what a run
            trains on.

    Returns:
        ``(images, poses, [H, W, focal], image_paths)`` with ``images`` ``(N, H, W, C)`` float32 in
        ``[0, 1]`` and ``poses`` ``(N, 4, 4)`` float32.
    """
    with open(os.path.join(basedir, f"transforms_{split}.json"), encoding="utf-8") as handle:
        meta = json.load(handle)

    poses: list[np.ndarray] = []
    images: list[np.ndarray] = []
    image_paths: list[str] = []

    for i, frame in enumerate(meta["frames"]):
        file_path = frame["file_path"]
        if not os.path.splitext(file_path)[1]:
            file_path = file_path + ".png"
        img_path = os.path.abspath(os.path.join(basedir, file_path))
        poses.append(np.array(frame["transform_matrix"]))
        image_paths.append(img_path)

        if read_offline or i == 0:
            img = imageio.imread(img_path)
            height, width = img.shape[:2]
            if factor > 1:
                img = Image.fromarray(img).resize((width // factor, height // factor))
            images.append((np.array(img) / 255.0).astype(np.float32))

    poses_arr = np.array(poses).astype(np.float32)
    images_arr = np.array(images).astype(np.float32)

    height, width = images_arr[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)

    return images_arr, poses_arr, [height, width, focal], image_paths


def load_meta_data(
    args: ConfigNode, mode: str = "train"
) -> tuple[Tensor, Tensor, Tensor, int, int, float, float, list[str]]:
    """Load one split of a scene as tensors, ready for ray generation.

    Only ``dataset.type: "synthetic"`` is reachable from the nine shipped configs, so that is the
    only branch implemented.

    Returns:
        ``(images, poses, masks, H, W, focal_x, focal_y, image_paths)``. ``images`` is
        ``(N, H, W, 3)`` already composited over ``bg_color`` when ``white_bg`` is set; ``masks`` is
        ``(N, H, W, 1)`` -- the raw alpha channel, *not* thresholded.
    """
    if args.type != "synthetic":
        raise ValueError(
            f"Unknown dataset type {args.type!r}. This release ships the Blender/transforms-JSON "
            "loader only ('synthetic'), which covers configs/nerfsyn/* and configs/editing/*."
        )

    images, poses, hwf, image_paths = load_blender_data(
        args.path, split=mode, factor=args.factor, read_offline=args.read_offline
    )
    height, width, focal = hwf
    focal_x, focal_y = focal, focal

    if images.shape[-1] == 4:
        masks = images[..., -1:]
        if args.white_bg:
            background = np.asarray(args.bg_color).reshape(1, 1, 1, 3)
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:]) * background
        else:
            images = images[..., :3]
    else:
        masks = np.ones((images.shape[0], images.shape[1], images.shape[2], 1), dtype=np.float32)

    return (
        torch.from_numpy(images).float(),
        torch.from_numpy(poses).float(),
        torch.from_numpy(masks).float(),
        height,
        width,
        focal_x,
        focal_y,
        image_paths,
    )

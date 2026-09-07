from __future__ import annotations

import glob
import os
import re

import imageio.v2 as imageio
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from dataset.load_nerfsyn import load_meta_data
from dataset.rays import build_intrinsics, generate_rays, pixel_centers, scale_pose, slice_patch
from utils.geometry import ray_sphere_intersection
from utils.ply import write_ply_points
from utils.runtime import ConfigNode

__all__ = [
    "DEPTH_EXTENSIONS",
    "RINDataset",
    "find_depth_file",
    "resolve_depth_split_dir",
]

#: Depth-map extensions, in the priority order they are probed.
DEPTH_EXTENSIONS = (".npy", ".tiff", ".png")

#: What upstream 2D Gaussian Splatting names its rendered depth maps.
_DEPTH_STEM = "depth_{index:05d}"


def resolve_depth_split_dir(gt_depth_dir: str, mode: str) -> str:
    """Point a NeRF-synthetic depth directory at the split actually being loaded.

    The 2DGS output tree is rendered once per scene under ``.../<split>/ours_30000/vis``, and the
    configs all record the ``test`` render. Training needs the ``train`` render from the same tree,
    so the split component is rewritten to ``mode``.

    Only paths containing ``nerf_synthetic`` are rewritten, which is what makes the Sketchfab
    editing scenes -- whose depth lives under a single ``test/`` directory used for both splits --
    keep the directory their config names. That asymmetry is load-bearing, not an oversight.

    The rewrite is anchored to whole path components rather than a raw substring replace, so a user
    directory named ``2dgs_pretrained/`` or ``constrained/`` is not silently mangled; the result is
    identical for every shipped config and safe for everything else.
    """
    if "nerf_synthetic" not in gt_depth_dir:
        return gt_depth_dir
    parts = gt_depth_dir.split(os.sep)
    return os.sep.join(mode if part in ("train", "test") else part for part in parts)


def find_depth_file(depth_dir: str, index: int) -> str | None:
    """Locate the depth map for frame ``index``, or ``None`` if the directory has none.

    Upstream 2DGS writes ``depth_%05d.tiff``, so that exact name is tried first for each extension.
    The fallback globs ``*<index>*<ext>`` and accepts the first sorted hit whose filename contains
    ``index`` as a whole run of digits. The digit-membership test is what stops frame 1 from
    matching ``depth_00021``; the glob alone would not.
    """
    for ext in DEPTH_EXTENSIONS:
        exact = os.path.join(depth_dir, _DEPTH_STEM.format(index=index) + ext)
        if os.path.isfile(exact):
            return exact
        for path in sorted(glob.glob(os.path.join(depth_dir, f"*{index}*{ext}"))):
            digits = [int(run) for run in re.findall(r"\d+", os.path.basename(path))]
            if index in digits:
                return path
    return None


def _missing_depth_message(depth_dir: str, index: int) -> str:
    return (
        f"No ground-truth depth map for frame {index} in {depth_dir!r}.\n"
        "\n"
        "dataset.load_gt_depth is true, so every frame of this split needs one. A missing map "
        "cannot be zero-filled: depth 0 back-projects onto the camera centre, and the "
        "surface-point loss would then pull the point cloud into the camera.\n"
        "\n"
        "Render the maps with 2D Gaussian Splatting, from its repository\n"
        "(https://github.com/hbb1/2d-gaussian-splatting):\n"
        "    python train.py -s <scene> -m <out> --eval -w\n"
        "    python render.py -m <out> -s <scene> --skip_train --skip_mesh\n"
        "\n"
        "The maps land in <out>/test/ours_30000/vis as depth_00000.tiff, depth_00001.tiff, ...; "
        "point dataset.gt_depth_dir at that directory. Set dataset.load_gt_depth: false to train "
        "without depth supervision."
    )


class RINDataset(Dataset):
    """Ray/Image/Normal dataset: one split of one scene, resident on ``device``.

    Args:
        args: The ``dataset`` config node (``eval.dataset`` merged over it, for the eval split).
        mode: ``"train"`` or ``"test"``. Selects the transforms file, and gates GT depth loading.
        device: Where the tensors live. Everything is moved eagerly, so this is the training
            device, not a CPU staging area.
    """

    def __init__(
        self, args: ConfigNode, mode: str = "train", device: str | torch.device = "cuda"
    ) -> None:
        self.args = args
        self.mode = mode

        if args.get("load_gt_sp", False):
            raise NotImplementedError(
                "dataset.load_gt_sp is not part of this release: it read a Blender-rendered "
                "surface-point/normal buffer from a hardcoded absolute path that no shipped "
                "config or published checkpoint uses. Use dataset.load_gt_depth instead."
            )

        images, c2w, masks, height, width, focal_x, focal_y, image_paths = load_meta_data(
            args, mode=mode
        )
        num_imgs = len(image_paths)

        self.num_imgs = num_imgs
        coord_scale = args.coord_scale
        if coord_scale != 1:
            c2w = scale_pose(c2w, coord_scale)

        self.device = device
        self.num_rays = height * width * num_imgs

        self.H = height
        self.W = width
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.cx = width / 2
        self.cy = height / 2
        self.c2w = c2w.to(device)
        self.image_paths = image_paths
        self.images = images.to(device)
        self.masks = masks.to(device)
        self.pix_coords = pixel_centers(height, width).to(device)
        self.intrinsics = build_intrinsics(focal_x, focal_y, height, width)

        self.foreground_coords = torch.nonzero(masks.squeeze(-1) > 0.5)

        rays = generate_rays(height, width, focal_x, focal_y, c2w)
        self.rayo = rays.origins.to(device)
        self.rayd = rays.directions.to(device)
        self.rays_d_no_norm = rays.raw_directions.to(device)
        self.pixels = rays.camera_plane.to(device)

        self.gt_depths: Tensor | None = None
        self.gt_surface_points: Tensor | None = None
        if args.load_gt_depth and mode == "train":
            self.gt_depths, self.gt_surface_points = self._load_gt_depth(rays.raw_directions)


    def _load_gt_depth(self, rays_d_no_norm: Tensor) -> tuple[Tensor, Tensor]:
        """Back-project the 2DGS depth maps into world-space surface points.

        Two subtleties, both preserved exactly:

        * The back-projection rides ``rays_d_no_norm`` -- the *un-normalised* direction -- and
          divides the depth by ``coord_scale``. The depth was multiplied by ``coord_scale`` a few
          lines earlier, so the two cancel: the product is a z-depth along a ``z = -1`` ray, and the
          scaling only ever lived in the stored depth map. Using ``rayd`` here instead would
          reinterpret a z-depth as a path length and bend every surface point toward the image
          edges.
        * Background pixels are replaced by the ray's exit point on the background sphere, keyed off
          the **ground-truth alpha mask**, not off ``depth == 0``. Most empty pixels do read zero,
          but a few (0.5% on dress frame 0) carry a 2DGS floater whose depth is several times the
          object's; a ``depth == 0`` test would keep those and scatter surface points far behind
          the scene. The alpha mask is authoritative about what is background, and the depth map
          is not.

        Returns:
            ``(depths, surface_points)`` of shapes ``(N, H, W)`` and ``(N, H, W, 3)``, both on
            ``self.device``.
        """
        args = self.args
        coord_scale = args.coord_scale
        depth_dir = resolve_depth_split_dir(args.gt_depth_dir, self.mode)

        radius_scale = args.get("bkg_sphere_radius")
        if radius_scale is None:
            raise ValueError(
                "dataset.bkg_sphere_radius is missing. The launcher copies "
                "geoms.points.bkg_sphere_radius into the dataset config before building the "
                "dataset, so that the background sphere the depth loader substitutes and the one "
                "the renderer attends to cannot drift apart."
            )
        sphere_radius = radius_scale * coord_scale
        sphere_center = [c * coord_scale for c in args.get("bkg_sphere_center", [0.0, 0.0, 0.0])]

        depths: list[Tensor] = []
        surface_points: list[Tensor] = []
        for index in range(self.num_imgs):
            path = find_depth_file(depth_dir, index)
            if path is None:
                raise FileNotFoundError(_missing_depth_message(depth_dir, index))

            if path.endswith(".npy"):
                raw = np.load(path)
            else:
                raw = imageio.imread(path)
            if raw.ndim == 3:
                raw = raw[..., 0]

            depth = torch.from_numpy(raw).float()
            if depth.shape[0] != self.H or depth.shape[1] != self.W:
                depth = F.interpolate(
                    depth[None, None], size=(self.H, self.W), mode="bilinear", align_corners=False
                ).squeeze()
            if coord_scale != 1:
                depth = depth * coord_scale

            ray_o = self.rayo[index].cpu()
            ray_d = rays_d_no_norm[index].cpu()
            points = ray_o + ray_d * depth.unsqueeze(-1) / coord_scale

            fg_mask = self.masks[index].cpu().squeeze() > 0.5
            bkg_mask = ~fg_mask
            if bkg_mask.any():
                bkg_points = ray_sphere_intersection(ray_o, ray_d, sphere_center, sphere_radius)
                points[bkg_mask] = bkg_points[bkg_mask]
                ray_d_len = torch.norm(ray_d, dim=-1)
                t_sphere = torch.norm(bkg_points - ray_o, dim=-1)
                bkg_depth = (t_sphere / ray_d_len) * coord_scale
                depth[bkg_mask] = bkg_depth[bkg_mask]

            if args.save_gt_sp:
                self._save_point_cloud(points, depth_dir, index)

            depths.append(depth)
            surface_points.append(points.to(self.device))

        return torch.stack(depths).to(self.device), torch.stack(surface_points)

    @staticmethod
    def _save_point_cloud(points: Tensor, out_dir: str, index: int) -> None:
        """Dump one frame's back-projected points as a PLY, for eyeballing depth alignment.

        This is written with :mod:`utils.ply` rather than open3d because open3d 0.18 against
        numpy >= 2 *segfaults* inside a live training process -- reproducibly, at the
        ``Vector3dVector`` conversion, once the torch stack is resident. The PLY is a debug
        artifact that never feeds back into training, so this writer (``double`` x/y/z,
        binary little-endian) changes nothing that a run computes.
        """
        write_ply_points(os.path.join(out_dir, f"frame_{index:04d}.ply"), points.reshape(-1, 3))


    def __len__(self) -> int:
        return self.num_imgs

    def __getitem__(self, idx: int) -> dict[str, object]:
        """One training sample: a random patch of frame ``idx``, or the whole frame.

        ``pix_coords`` is the only entry that is not per-frame; it is the same grid every time,
        cropped to the patch.
        """
        if not self.args.extract_patch:
            sample = {
                "idx": idx,
                "patch_idx": 0,
                "image": self.images[idx],
                "mask": self.masks[idx],
                "rayd": self.rayd[idx],
                "rays_d_no_norm": self.rays_d_no_norm[idx],
                "rayo": self.rayo[idx],
                "pix_coords": self.pix_coords,
                "pixels": self.pixels[idx],
                "intrinsics": self.intrinsics,
            }
            if self.gt_depths is not None:
                sample["depth"] = self.gt_depths[idx]
                sample["surface_points"] = self.gt_surface_points[idx]
            return sample

        patch_h = self.args.patches.height
        patch_w = self.args.patches.width
        top = np.random.randint(0, self.H - patch_h)
        left = np.random.randint(0, self.W - patch_w)

        sample = {
            "idx": idx,
            "patch_idx": 0,
            "image": slice_patch(self.images[idx], top, left, patch_h, patch_w),
            "mask": slice_patch(self.masks[idx], top, left, patch_h, patch_w),
            "rayd": slice_patch(self.rayd[idx], top, left, patch_h, patch_w),
            "rays_d_no_norm": slice_patch(
                self.rays_d_no_norm[idx], top, left, patch_h, patch_w
            ),
            "rayo": self.rayo[idx],
            "pix_coords": slice_patch(self.pix_coords, top, left, patch_h, patch_w),
            "pixels": slice_patch(self.pixels[idx], top, left, patch_h, patch_w),
            "intrinsics": self.intrinsics,
        }
        if self.gt_depths is not None:
            sample["depth"] = self.gt_depths[idx][top : top + patch_h, left : left + patch_w]
            sample["surface_points"] = slice_patch(
                self.gt_surface_points[idx], top, left, patch_h, patch_w
            )
        return sample

    def get_full_img(self, img_idx: int) -> dict[str, object]:
        """A whole frame with a leading batch dimension, for validation and test renders.

        ``surface_points`` is deliberately absent even when depth is loaded -- since it is never
        emitted here, the eval split can skip loading depth entirely. ``pix_coords`` keeps its
        unbatched ``(H, W, 2)`` shape here, unlike every other entry; callers index it without a
        batch axis.
        """
        sample = {
            "idx": img_idx,
            "patch_idx": 0,
            "image": self.images[img_idx][None],
            "mask": self.masks[img_idx][None],
            "rayd": self.rayd[img_idx][None],
            "rays_d_no_norm": self.rays_d_no_norm[img_idx][None],
            "rayo": self.rayo[img_idx][None],
            "pix_coords": self.pix_coords,
            "pixels": self.pixels[img_idx][None],
        }
        if self.gt_depths is not None:
            sample["depth"] = self.gt_depths[img_idx][None]
        return sample

    def get_c2w(self, img_idx: int | list[int] | Tensor) -> Tensor:
        """The pose(s) of ``img_idx``, always batched as ``(B, 4, 4)``."""
        selected = self.c2w[img_idx]
        if selected.dim() == 2:
            selected = selected.unsqueeze(0)
        return selected

    def get_new_rays(self, c2w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Rays for an arbitrary pose at this scene's resolution and intrinsics.

        Used to render novel views (interpolated or edited camera paths) that are not in the split.

        Returns:
            ``(origins, unit directions, camera-frame rays)`` -- ``rays_d_no_norm`` is dropped
            because novel-view rendering never back-projects a depth.
        """
        rays = generate_rays(self.H, self.W, self.focal_x, self.focal_y, c2w)
        return rays.origins, rays.directions, rays.camera_plane

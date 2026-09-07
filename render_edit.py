from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from dataset import get_dataset
from edit.pcd import find_sequence_dir, list_point_clouds, load_deformed_points
from edit.render import render_edited_view, to_uint8
from models import get_model
from utils.config import ROOT, ResolvedConfig, load_config
from utils.runtime import ConfigNode, to_config_node
from utils.session import Logger, setup_seed, setup_torch
from uv import get_nuvo
from uv.bbox import load_bounding_box
from uv.texture import (
    create_edit_mask_from_texture_maps,
    grid_layout,
    tile_image_to_charts,
    unpack_grid_to_texture_maps,
    validate_grid_image,
)

#: Filenames the stage-b checkpoint is looked for under, in order.
NUVO_CHECKPOINT_NAMES = (
    Path("nuvo_model_with_texture.ckpt"),
    Path("texture_training") / "texture_final.ckpt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a PointGT geometry + texture edit")
    parser.add_argument("--opt", type=str, required=True,
                        help="Scene *_uv config (YAML): renderer settings plus the nuvo block")
    parser.add_argument("--texture", type=str, required=True,
                        help="Edited texture image; tiled across the atlas charts unless "
                             "--texture_is_grid")
    parser.add_argument("--texture_is_grid", action="store_true",
                        help="Treat --texture as an already-packed chart grid, as exported by "
                             "train_uv.py or the interactive editor")
    parser.add_argument("--texture_grid_columns", type=int, default=0,
                        help="Columns in that grid; 0 (default) uses the layout the atlas's chart "
                             "count implies (4 charts -> one row of 4), which is what "
                             "train_uv.py writes")
    parser.add_argument("--nuvo_ckpt", type=str, default=None,
                        help="Stage-b atlas checkpoint; defaults to nuvo_model_with_texture.ckpt "
                             "beside the config, then beside the stage-a checkpoint")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Stage-a PAPR .pth; overrides the config's load_path")
    parser.add_argument("--pcd_dir", type=str, default=None,
                        help="Directory of deformed point clouds; defaults to rbf_pcds_fixcp, "
                             "else rbf_pcds, beside the config and then beside the checkpoint")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--view", type=int, default=0,
                        help="Test-split camera index the render starts from")
    parser.add_argument("--frames", type=int, default=None,
                        help="Frames to render; defaults to every point cloud found")
    parser.add_argument("--fps", type=int, default=24, help="Frame rate of the optional video")
    parser.add_argument("--video", action="store_true",
                        help="Also encode video.mp4 beside the frames (needs ffmpeg)")
    parser.add_argument("--mask_threshold", type=float, default=0.01,
                        help="L2 texel distance above which a texel counts as edited")
    return parser.parse_args()


def _resolve_relative(candidate: str) -> Path:
    """A command-line path, tried as typed and then against the release root."""
    path = Path(candidate).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def resolve_checkpoint(values: ConfigNode, override: str | None) -> Path:
    """The stage-a ``.pth``: ``--load_path`` beats the config's ``load_path``."""
    candidate = override or str(values.get("load_path", "") or "")
    if not candidate:
        raise FileNotFoundError(
            "No stage-a checkpoint. Set load_path in the config or pass --load_path."
        )
    path = _resolve_relative(candidate)
    if path.suffix != ".pth":
        path = path / "model.pth"
    if not path.is_file():
        raise FileNotFoundError(f"No stage-a checkpoint at {path}")
    return path


def resolve_nuvo_checkpoint(config_dir: Path, checkpoint_dir: Path, override: str | None) -> Path:
    """The stage-b atlas checkpoint, searched beside the config and then beside the checkpoint."""
    if override:
        path = _resolve_relative(override)
        if not path.is_file():
            raise FileNotFoundError(f"No atlas checkpoint at {path}")
        return path
    tried = [base / name for base in (config_dir, checkpoint_dir) for name in
             NUVO_CHECKPOINT_NAMES]
    for candidate in tried:
        if candidate.is_file():
            return candidate
    listing = "\n  ".join(str(p) for p in tried)
    raise FileNotFoundError(
        "No stage-b atlas checkpoint found. Pass --nuvo_ckpt, or put one at any of:\n  " + listing
    )


def resolve_sequence_dir(config_dir: Path, checkpoint_dir: Path, override: str | None) -> Path:
    """The deformed point-cloud directory, searched the same way."""
    if override:
        path = _resolve_relative(override)
        if not path.is_dir():
            raise NotADirectoryError(f"No deformed point-cloud directory at {path}")
        return path
    for base in (config_dir, checkpoint_dir):
        found = find_sequence_dir(base)
        if found is not None:
            return found
    raise FileNotFoundError(
        "No deformed point-cloud directory found. Pass --pcd_dir, or put an 'rbf_pcds' (or "
        f"'rbf_pcds_fixcp') directory under {config_dir} or {checkpoint_dir}."
    )


def dataset_config(values: ConfigNode) -> ConfigNode:
    """The ``dataset`` block with the background sphere copied in.

    Two things happen here, and both matter. The sphere's radius falls back to
    ``bkg_points_init_scale[0]`` when the config does not
    name it -- inert while ``configs/default*.yml`` supply ``bkg_sphere_radius: 5.0``, kept because
    a scene config that strips the defaults would otherwise get a *different* sphere in the dataset
    than in the renderer. And the resolved values are mirrored onto the dataset block, which is
    where :class:`~dataset.dataset.RINDataset` reads them to place a background pixel's surface
    point when it loads GT depth. No config file ships them under ``dataset``; without this copy
    that lookup fails outright.
    """
    point_opt = values.geoms.points
    radius = point_opt.get("bkg_sphere_radius", None)
    if radius is None:
        radius = point_opt.bkg_points_init_scale[0]
    center = point_opt.get("bkg_sphere_center", None)
    if center is None:
        center = [0.0, 0.0, 0.0]

    merged = values.dataset.to_dict()
    merged["bkg_sphere_radius"] = radius
    merged["bkg_sphere_center"] = list(center)
    return to_config_node(merged, "dataset")


def render_nuvo_config(nuvo: ConfigNode) -> ConfigNode:
    """The ``nuvo`` config with the *render-time* texture padding forced to reflection.

    The archived renderer set ``nuvo_conf.texture.padding_mode = "reflection"`` on the live model
    immediately before sampling, overriding the
    ``"zeros"`` every config carries. It matters: a canonical surface point can land a fraction of
    a texel outside its chart's UV square, and zero padding turns that into a black rim along every
    chart seam, while reflection continues the texture across it. Training deliberately keeps
    ``"zeros"`` -- an out-of-square UV should cost the atlas something -- so this is a render-time
    override, applied to a copy of the config rather than by mutating the built model.
    """
    conf = nuvo.to_dict()
    conf["texture"]["padding_mode"] = "reflection"
    return to_config_node(conf, "nuvo")


def atlas_state_dict(payload: Any) -> Any:
    """Unwrap whichever container a stage-b checkpoint used.

    Three shapes are in circulation, all of them archived: ``{"step": ..., "model_state_dict":
    ...}`` from the texture stage, ``{step: state_dict}`` from the PAPR-style saver, and a bare
    state dict.
    """
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    if len(payload) == 1:
        only = next(iter(payload.values()))
        if isinstance(only, dict):
            return only
    return payload


class NuvoTextureAtlas:
    """:class:`edit.render.TextureAtlas` over :class:`uv.nuvo.Nuvo`.

    Exists so :mod:`edit.render` never sees the atlas's six-argument sampling signature, three of
    whose arguments (``rays_o``, ``rays_d``, ``normals``) only the view-dependent texture-MLP path
    reads and no shipped config enables.
    """

    def __init__(self, nuvo: Any) -> None:
        self._nuvo = nuvo
        self.num_charts = int(nuvo.num_charts)
        self.chart_resolution = int(nuvo.texture_map_res_per_chart)
        self._argmax = bool(nuvo.conf.texture.eval_argmax)

    @property
    def texture_map(self) -> Tensor:
        """The atlas's own learned map, activated.

        This is the reference the edit mask is measured against.
        """
        with torch.no_grad():
            return self._nuvo.texture_map_act(self._nuvo.texture_map).detach()

    def sample_texture(self, points: Tensor, texture_map: Tensor) -> Tensor:
        zeros = torch.zeros_like(points)
        rgb, _, _ = self._nuvo.get_rgb_from_texture_map_for_points(
            points, zeros, zeros, normals=None, argmax=self._argmax,
            render_checkboard=False, pred_uvs=None, pred_chart_probs=None,
            texture_map=texture_map, return_additional_texture_map=False,
        )
        return rgb


def load_atlas(values: ConfigNode, checkpoint: Path, device: torch.device) -> NuvoTextureAtlas:
    """Build stage b's atlas and restore it from ``checkpoint``."""
    nuvo = get_nuvo(render_nuvo_config(values.nuvo), device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    skipped = nuvo.load_model(atlas_state_dict(payload))
    if "texture_map" in skipped:
        raise ValueError(
            f"{checkpoint} did not supply a texture map that fits this config. The atlas was "
            f"built at nuvo.texture.texture_map_res={values.nuvo.texture.texture_map_res} "
            f"({nuvo.texture_map_res_per_chart} per chart, {nuvo.num_charts} charts); the "
            "checkpoint's is a different shape. Render with the config the atlas was trained "
            "under."
        )
    if skipped:
        print(f"[edit] atlas: {len(skipped)} checkpoint entries skipped: {sorted(skipped)}")
    nuvo.eval()
    return NuvoTextureAtlas(nuvo)


def prepare_texture(
    path: str | Path,
    *,
    num_charts: int,
    chart_resolution: int,
    reference: Tensor,
    is_grid: bool,
    columns: int | None,
    device: torch.device,
) -> tuple[Tensor, int]:
    """Read ``--texture`` into a ``(num_charts, H, W, 3)`` map shaped like ``reference``.

    ``--texture_is_grid`` is checked against the layout rather than guessed at: the atlas's chart
    count fixes the layout (:func:`uv.texture.grid_layout`), and a grid whose pixels are not that
    layout is rejected by :func:`uv.texture.validate_grid_image` instead of being unpacked into
    scrambled charts. ``columns`` (``--texture_grid_columns``) overrides the expected layout for a
    grid packed some other way, and is validated the same way.

    Returns the map and the grid column count that was used, for the log line.
    """
    image = np.array(Image.open(path).convert("RGB"))
    if is_grid:
        _, columns = validate_grid_image(image.shape, num_charts, columns)
    else:
        _, columns = grid_layout(num_charts, columns)
        image = tile_image_to_charts(image, num_charts, chart_resolution, columns=columns)
    charts = unpack_grid_to_texture_maps(image, num_charts, columns=columns)
    texture = torch.as_tensor(charts, dtype=torch.float32, device=device)

    target = tuple(reference.shape[1:3])
    if tuple(texture.shape[1:3]) != target:
        texture = F.interpolate(
            texture.permute(0, 3, 1, 2), size=target, mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
    return texture, columns


def rotation_z(angle: float, device: torch.device) -> Tensor:
    """``(4, 4)`` rotation about the world Z axis.

    Applied on the *left* of a camera-to-world matrix, so it rotates the camera around the world
    origin rather than spinning it in place. Z is the up axis of the Blender-convention scenes
    these edits are authored in. A pure rotation commutes with the ``coord_scale`` prefactor
    already baked into ``dataset.c2w``, so no unscaling is needed here.
    """
    cos, sin = math.cos(angle), math.sin(angle)
    matrix = torch.eye(4, dtype=torch.float32, device=device)
    matrix[0, 0], matrix[0, 1] = cos, -sin
    matrix[1, 0], matrix[1, 1] = sin, cos
    return matrix


def run(cli: argparse.Namespace, config: ResolvedConfig, out_dir: Path) -> None:
    """Load everything, render every frame in both camera modes, write the results."""
    values = config.values
    config_dir = config.source.parent
    if "nuvo" not in values:
        raise ValueError(
            f"{config.source} has no `nuvo:` block, so there is no atlas to render through. "
            "--opt takes the scene's *_uv config (configs/editing/<scene>_uv.yml), which carries "
            "the renderer's settings and the atlas's together."
        )

    setup_seed(values.seed)
    device = setup_torch(values)

    checkpoint = resolve_checkpoint(values, cli.load_path)
    model = get_model(values, device).to(device)
    report = model.load(checkpoint)
    model.eval()

    nuvo_checkpoint = resolve_nuvo_checkpoint(config_dir, checkpoint.parent, cli.nuvo_ckpt)
    sequence_dir = resolve_sequence_dir(config_dir, checkpoint.parent, cli.pcd_dir)

    dataset = get_dataset(dataset_config(values), "test", device)
    model.set_camera(dataset.focal_x, dataset.focal_y, dataset.cx, dataset.cy, dataset.H,
                     dataset.W)

    atlas = load_atlas(values, nuvo_checkpoint, device)
    original_texture = atlas.texture_map
    texture, columns = prepare_texture(
        cli.texture, num_charts=atlas.num_charts, chart_resolution=atlas.chart_resolution,
        reference=original_texture, is_grid=cli.texture_is_grid,
        columns=cli.texture_grid_columns or None, device=device,
    )
    mask_texture = create_edit_mask_from_texture_maps(original_texture, texture,
                                                     threshold=cli.mask_threshold)

    paths = list_point_clouds(sequence_dir)
    if not paths:
        raise FileNotFoundError(f"No point clouds in {sequence_dir}")
    if cli.frames is not None:
        if cli.frames < 1:
            raise ValueError(f"--frames must be at least 1, got {cli.frames}")
        paths = paths[:cli.frames]

    if not 0 <= cli.view < dataset.num_imgs:
        raise ValueError(f"--view {cli.view} is outside [0, {dataset.num_imgs - 1}]")

    bbox_path = values.nuvo.sample.get("bounding_box_mesh_path", None)
    bbox_hull = load_bounding_box(bbox_path, coord_scale=1.0) if bbox_path else None

    canonical_points = model.points.detach().clone()
    coord_scale = float(model.coord_scale)
    base_c2w = dataset.get_c2w(cli.view)[0].to(device).float()

    print(f"[edit] config     {config.source} (sha256 {config.config_sha256[:12]})")
    print(f"[edit] papr       {report.path} (step {report.step}, {report.n_points} points)")
    print(f"[edit] atlas      {nuvo_checkpoint} ({atlas.num_charts} charts, "
          f"texture {tuple(original_texture.shape[1:3])})")
    grid_rows = grid_layout(atlas.num_charts, columns)[0]
    layout = (f"packed grid, {grid_rows} row(s) x {columns} column(s)" if cli.texture_is_grid
              else f"tiled across {atlas.num_charts} charts at {atlas.chart_resolution}px")
    print(f"[edit] texture    {cli.texture} ({layout}); "
          f"{int(mask_texture.sum().item())} of {mask_texture.numel()} texels edited")
    print(f"[edit] geometry   {sequence_dir} ({len(paths)} frames)")
    print(f"[edit] camera     view {cli.view}, {dataset.H}x{dataset.W}, tiles "
          f"{values.test.max_height}x{values.test.max_width}")

    modes = {"orbit": out_dir / "orbit", "fixed": out_dir / "fixed"}
    for directory in modes.values():
        (directory / "frames").mkdir(parents=True, exist_ok=True)
    written: dict[str, list[np.ndarray]] = {name: [] for name in modes}

    for index, path in enumerate(paths):
        deformed = load_deformed_points(
            path, expected_count=canonical_points.shape[0], coord_scale=coord_scale, device=device,
        )
        orbit_c2w = rotation_z(2.0 * math.pi * index / len(paths), device) @ base_c2w
        for name, c2w in (("orbit", orbit_c2w), ("fixed", base_c2w)):
            frame = render_edited_view(
                model, dataset,
                atlas=atlas,
                texture_map=texture,
                mask_texture_map=mask_texture,
                deformed_points=deformed,
                canonical_points=canonical_points,
                c2w=c2w,
                max_height=int(values.test.max_height),
                max_width=int(values.test.max_width),
                bbox_hull=bbox_hull,
                step=report.step,
            )
            image = to_uint8(frame.final_rgb)
            imageio.imwrite(modes[name] / "frames" / f"{index:04d}.png", image)
            written[name].append(image)
        print(f"  frame {index:04d}/{len(paths) - 1}  {path.name}", flush=True)

    for name, directory in modes.items():
        print(f"[edit] {name}: {len(written[name])} frames -> {directory / 'frames'}")
        if cli.video:
            video = directory / "video.mp4"
            imageio.mimwrite(video, np.stack(written[name]), fps=cli.fps, quality=9)
            print(f"[edit] {name}: wrote {video}")


def main() -> None:
    cli = parse_args()
    config = load_config(cli.opt)
    out_dir = Path(cli.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = Logger(out_dir / "render_edit.log", saved_out)
    sys.stderr = Logger(out_dir / "render_edit_error.log", saved_err)
    try:
        run(cli, config, out_dir)
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout, sys.stderr = saved_out, saved_err


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import imageio.v2 as imageio
import numpy as np
import torch
from torch import Tensor

from dataset import get_dataset, get_loader
from models import get_model
from models.surface import get_surface_points
from utils.config import ROOT, ResolvedConfig, deep_merge, load_config
from utils.metrics import METRICS_FILENAME, LpipsMetric, MetricAggregator
from utils.ply import write_ply_points
from utils.runtime import ConfigNode, to_config_node
from utils.session import Logger, archive_config, make_run_dir, setup_seed, setup_torch

#: The perceptual backbone the published LPIPS column was computed with. Fixed rather than exposed
#: as a flag: the marker-directory field is literally named ``LPIPSv``, and a sidecar that could
#: hold either backbone under the same key is how the mixed-row table happened.
LPIPS_BACKBONE = "vgg"

#: Surface-point fusion used by ``--save_sp``. ``position-sphere`` places a background ray's
#: "surface" on the background sphere rather than at the weighted mean of its (irrelevant)
#: foreground candidates.
SURFACE_FUSE_TYPE = "position-sphere"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and score a PointGT test split")
    parser.add_argument("--opt", type=str, required=True, help="Scene config (YAML)")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Checkpoint .pth to evaluate; overrides test.load_path")
    parser.add_argument("--name", type=str, default="",
                        help="Suffix appended to the output directory name")
    parser.add_argument("--mask", action="store_true",
                        help="Mask the background out (over white) with the GT alpha before "
                             "scoring")
    parser.add_argument("--save_sp", action="store_true",
                        help="Also dump each frame's fused surface points as a .ply")
    return parser.parse_args()


def resolve_checkpoint(values: ConfigNode, run_dir: Path, override: str | None) -> Path:
    """Decide which ``.pth`` to score.

    Precedence is ``--load_path`` > ``test.load_path`` > ``<run_dir>/model.pth``, the last being
    where a finished training run leaves its checkpoint. A path without a ``.pth`` suffix is taken
    to be a directory holding ``model.pth`` -- that is how the archived ``PAPR.save`` directories
    (``model.pth`` beside ``optimizers.pth``) are addressed.
    """
    candidate = override or values.test.get("load_path", "") or ""
    path = Path(candidate) if candidate else run_dir / "model.pth"
    if path.suffix != ".pth":
        path = path / "model.pth"
    if not path.is_absolute():
        path = path if path.exists() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Pass --load_path, set test.load_path in the config, or "
            f"train first so that {run_dir / 'model.pth'} exists."
        )
    return path


def split_config(values: ConfigNode, entry: ConfigNode) -> ConfigNode:
    """One ``test.datasets`` entry merged over the scene's ``dataset`` block.

    ``geoms.points.bkg_sphere_radius`` / ``bkg_sphere_center`` are copied in because
    :class:`~dataset.dataset.RINDataset` substitutes the ray's exit point on *that* sphere for
    background pixels when it loads GT depth, and a dataset sphere that disagreed with the one the
    renderer attends to would silently mis-place every background surface point. Copied
    unconditionally even though a test split never loads depth -- so the two cannot drift apart
    if that ever changes.
    """
    point_opt = values.geoms.points
    merged = deep_merge(values.dataset.to_dict(), entry.to_dict())
    merged["bkg_sphere_radius"] = point_opt.get("bkg_sphere_radius", 5.0)
    merged["bkg_sphere_center"] = point_opt.get("bkg_sphere_center", [0.0, 0.0, 0.0])
    return to_config_node(merged, "dataset")


def split_name(entry: ConfigNode, suffix: str) -> str:
    name = str(entry.get("name", "testset"))
    return f"{name}_{suffix}" if suffix else name


def tile_bounds(extent: int, tile: int) -> Iterator[tuple[int, int]]:
    """``[start, end)`` spans covering ``extent``, the last one short."""
    if tile <= 0:
        raise ValueError(f"tile size must be positive, got {tile}")
    for start in range(0, extent, tile):
        yield start, min(start + tile, extent)


@dataclass(frozen=True)
class FrameRender:
    """One scored frame: the image, plus what the depth map and ``--save_sp`` need."""

    #: ``(1, H, W, 3)`` float32 in ``[0, 1]``, after ``last_act``, clamp and the optional mask.
    rgb: Tensor
    #: ``(1, H, W, K+1, 1)`` attention weights, background slot last.
    attn: Tensor
    #: ``(1, H, W, K)`` indices into :attr:`points`.
    select_k_ind: Tensor
    #: ``(M, 3)`` the cloud those indices address.
    points: Tensor
    #: ``(1, H, W, 1, 3)`` where each ray leaves the background sphere.
    sphere_intersection: Tensor
    #: ``(1, H, W)`` attention-weighted depth in dataset units (already divided by coord_scale).
    depth: Tensor


@torch.no_grad()
def render_frame(
    model: Any,
    *,
    rays_o: Tensor,
    rays_d: Tensor,
    c2w: Tensor,
    pix_coords: Tensor,
    mask: Tensor,
    step: int,
    max_height: int,
    max_width: int,
    apply_mask: bool,
) -> FrameRender:
    """Tile-render one full frame and composite it.

    ``step`` is the checkpoint's training step and is **not** decorative: it drives the
    spherical-harmonic band schedule (``configs/nerfsyn/chair.yml`` is the ``use_sh`` config), so
    rendering a 250k-step checkpoint at ``step=-1`` would evaluate a different radiance field.
    """
    n, height, width, _ = rays_d.shape

    fused: Tensor | None = None
    attn: Tensor | None = None
    select_k_ind: Tensor | None = None
    sphere_intersection: Tensor | None = None
    points: Tensor | None = None

    for top, bottom in tile_bounds(height, max_height):
        for left, right in tile_bounds(width, max_width):
            out = model.evaluate(
                rays_o,
                rays_d[:, top:bottom, left:right],
                c2w,
                pix_coords[:, top:bottom, left:right],
                step=step,
            )
            if fused is None:
                slots = out.attn.shape[-2]
                candidates = out.select_k_ind.shape[-1]
                channels = out.fused_features.shape[-1]
                device = rays_d.device
                fused = torch.zeros(n, height, width, 1, channels, device=device)
                attn = torch.zeros(n, height, width, slots, 1, device=device)
                select_k_ind = torch.zeros(n, height, width, candidates, dtype=torch.long,
                                           device=device)
                sphere_intersection = torch.zeros(n, height, width, 1, 3, device=device)
            fused[:, top:bottom, left:right] = out.fused_features
            attn[:, top:bottom, left:right] = out.attn
            select_k_ind[:, top:bottom, left:right] = out.select_k_ind
            sphere_intersection[:, top:bottom, left:right] = out.sphere_intersection
            points = out.cloud.points

    if fused is None:  # pragma: no cover - a zero-sized frame cannot come out of the loader
        raise ValueError("frame has no tiles; check test.max_height / test.max_width")

    rgb, _ = model.decode(fused.squeeze(-2), n, height, width)

    if model.args.models.attn.get("append_bkg_points_alpha_blend", False):
        bkg_attn = attn[..., -1, :]
        bkg_color = (model.bkg_feats * model.bkg_scaler.squeeze(0)).expand(n, height, width, -1)
        rgb = rgb * (1 - bkg_attn) + bkg_color * bkg_attn

    rgb = torch.clamp(model.last_act(rgb), 0, 1).float()
    if apply_mask:
        rgb = rgb * mask + (1 - mask) * 1.0

    return FrameRender(
        rgb=rgb,
        attn=attn,
        select_k_ind=select_k_ind,
        points=points,
        sphere_intersection=sphere_intersection,
        depth=attention_depth(attn, points[select_k_ind], sphere_intersection, rays_o,
                              model.coord_scale),
    )


def attention_depth(
    attn: Tensor,
    selected_points: Tensor,
    sphere_intersection: Tensor,
    rays_o: Tensor,
    coord_scale: float,
) -> Tensor:
    """Attention-weighted depth per pixel, in dataset units.

    Each slot contributes its distance to the plane through the camera centre with normal
    ``-rays_o`` -- i.e. the depth is measured along the view axis, not along each ray, which is why
    it is computed from ``rays_o`` alone and not per-pixel. The background slot contributes its
    distance to the sphere intersection. The final division by ``coord_scale`` puts the result
    back in the dataset's own units.
    """
    axis = -rays_o
    offset = torch.sum(axis * rays_o)
    distances = torch.abs(torch.sum(selected_points * axis, -1) - offset) / torch.norm(axis)
    background = torch.norm(sphere_intersection.squeeze(-2) - rays_o, dim=-1, keepdim=True)
    distances = torch.cat([distances, background], dim=-1)
    return torch.sum(attn.squeeze(-1) * distances, dim=-1) / coord_scale


def _to_u8(image: Tensor) -> np.ndarray:
    return (image.detach().cpu().numpy() * 255).astype(np.uint8)


def _normalize(values: np.ndarray) -> np.ndarray:
    """Min-max a float map into ``[0, 1]`` for display.

    Guarded against a constant map, which would otherwise divide by zero. A visualisation,
    never a reported number.
    """
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return (values - low) / (high - low)


def save_frame_images(out_dir: Path, index: int, render: FrameRender) -> None:
    """Write one frame's PNGs: the render, its attention depth, and the background weight.

    Metrics live in ``metrics.json``, keyed by the same index, so the filenames here stay stable
    and sortable. A foreground-only RGB (identical to the render whenever the alpha blend is off,
    i.e. always), a min-perpendicular-distance map, and a point-density grid are not written here:
    those were debugging aids for the renderer, not outputs of an evaluation.
    """
    imageio.imwrite(out_dir / f"frame_{index:04d}_rgb.png", _to_u8(render.rgb.squeeze(0)))
    imageio.imwrite(
        out_dir / f"frame_{index:04d}_depth.png",
        (_normalize(render.depth.squeeze(0).detach().cpu().numpy().astype(np.float32)) * 255)
        .astype(np.uint8),
    )
    imageio.imwrite(
        out_dir / f"frame_{index:04d}_bkgmask.png",
        _to_u8(render.attn[..., -1, :].squeeze(0).squeeze(-1).clamp(0, 1)),
    )


def save_frame_surface_points(out_dir: Path, index: int, render: FrameRender, *, rays_o: Tensor,
                              rays_d: Tensor, coord_scale: float) -> Path:
    """Fuse each ray's attention into one surface point and dump the frame as a PLY.

    Lands in the run directory rather than in ``dataset.gt_depth_dir`` -- a directory a user
    reasonably believes is read-only input.
    """
    surface_points = get_surface_points(
        render.points,
        render.select_k_ind,
        rays_o,
        rays_d,
        render.attn,
        SURFACE_FUSE_TYPE,
        sphere_intersection=render.sphere_intersection,
        coord_scale=coord_scale,
    )
    return write_ply_points(out_dir / f"frame_{index:04d}_surface_points.ply",
                            surface_points.reshape(-1, 3))


def evaluate_split(
    model: Any,
    values: ConfigNode,
    entry: ConfigNode,
    *,
    run_dir: Path,
    device: torch.device,
    step: int,
    cli: argparse.Namespace,
) -> dict[str, Any]:
    """Render and score one ``test.datasets`` entry. Returns the metrics summary it wrote."""
    dataset_args = split_config(values, entry)
    mode = str(dataset_args.get("mode", "test"))
    name = split_name(entry, cli.name)

    dataset = get_dataset(dataset_args, mode, device)
    model.set_camera(dataset.focal_x, dataset.focal_y, dataset.cx, dataset.cy, dataset.H,
                     dataset.W)
    loader = get_loader(dataset, dataset_args, "test")

    out_dir = run_dir / "test" / f"images-step{step}" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregator = MetricAggregator(
        lpips_metric=LpipsMetric(LPIPS_BACKBONE, normalize=False, device=device)
    )
    save_fig = bool(values.test.get("save_fig", True))
    max_height = int(values.test.max_height)
    max_width = int(values.test.max_width)

    print(f"[test] {name}: {len(loader)} frames at {dataset.H}x{dataset.W}, "
          f"tiles {max_height}x{max_width} -> {out_dir}")

    for index, batch in enumerate(loader):
        rays_o = batch["rayo"].to(device)
        rays_d = batch["rayd"].to(device)
        image = batch["image"].to(device).float()
        mask = batch["mask"].to(device)
        pix_coords = batch["pix_coords"].to(device)
        c2w = dataset.get_c2w(batch["idx"]).to(device)

        render = render_frame(
            model,
            rays_o=rays_o,
            rays_d=rays_d,
            c2w=c2w,
            pix_coords=pix_coords,
            mask=mask,
            step=step,
            max_height=max_height,
            max_width=max_width,
            apply_mask=cli.mask,
        )

        frame = aggregator.update(render.rgb.permute(0, 3, 1, 2), image.permute(0, 3, 1, 2))[0]
        lpips_text = "n/a" if frame.lpips is None else f"{frame.lpips:.4f}"
        print(f"  frame {index:04d}/{len(loader) - 1}  psnr {frame.psnr:7.4f}  "
              f"ssim {frame.ssim:.4f}  lpips[{LPIPS_BACKBONE}] {lpips_text}", flush=True)

        if save_fig:
            save_frame_images(out_dir, index, render)
        if cli.save_sp:
            save_frame_surface_points(out_dir, index, render, rays_o=rays_o, rays_d=rays_d,
                                      coord_scale=model.coord_scale)

    aggregator.write_json(out_dir)
    summary = aggregator.summary()
    aggregate = summary["aggregate"]
    print(f"[test] {name}: {summary['n_frames']} frames | "
          f"PSNR {aggregate['psnr']:.4f} | SSIM {aggregate['ssim']:.4f} | "
          f"LPIPS[{LPIPS_BACKBONE}] {aggregate.get('lpips', float('nan')):.4f}")
    if summary["legacy_marker"]:
        print(f"[test] {name}: archived-marker form {summary['legacy_marker']}")
    print(f"[test] {name}: wrote {out_dir / METRICS_FILENAME}")
    return summary


def run(cli: argparse.Namespace, config: ResolvedConfig, run_dir: Path) -> None:
    """Build the model, load the checkpoint and score every ``test.datasets`` entry."""
    values = config.values
    archive_config(config, run_dir)

    setup_seed(values.seed)
    device = setup_torch(values)

    model = get_model(values, device)
    checkpoint = resolve_checkpoint(values, run_dir, cli.load_path)
    report = model.load(checkpoint)
    model.eval()

    print(f"[test] config     {config.source} (sha256 {config.config_sha256[:12]}, "
          f"fingerprint {config.fingerprint[:12]})")
    print(f"[test] checkpoint {report.path}")
    print(f"[test]            sha256 {report.sha256[:12]}, step {report.step}, "
          f"{report.n_points} points, container {report.source_schema!r}")
    if report.migrations or report.derived:
        print(f"[test]            migrations {list(report.migrations)}, "
              f"derived {list(report.derived)}")
    print(f"[test] masking    {'on (--mask)' if cli.mask else 'off'}")

    for entry in values.test.datasets:
        evaluate_split(model, values, entry, run_dir=run_dir, device=device, step=report.step,
                       cli=cli)


def main() -> None:
    cli = parse_args()
    config = load_config(cli.opt)
    run_dir = make_run_dir(config.values)

    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = Logger(run_dir / "test.log", saved_out)
    sys.stderr = Logger(run_dir / "test_error.log", saved_err)
    try:
        run(cli, config, run_dir)
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout, sys.stderr = saved_out, saved_err


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from dataset import get_dataset
from models import get_model
from models.optim import init_optimizers
from models.optim import step as optimizer_step
from utils.config import load_config
from utils.ply import write_ply_points
from utils.runtime import ConfigNode, to_config_node
from utils.session import (
    Logger,
    archive_config,
    make_run_dir,
    setup_seed,
    setup_torch,
    snapshot_code,
)
from uv.bbox import load_bounding_box
from uv.losses import geometry_losses, uv_range_loss
from uv.nuvo import Nuvo
from uv.sampling import SurfacePointSet, extract_surface_points, warmup_chart_assignment
from uv.texture import grid_layout, pack_texture_map_to_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PointGT stage-b UV atlas + texture training")
    parser.add_argument("--opt", type=str, required=True,
                        help="Path to a UV config (.yml) whose name ends in _uv, e.g. "
                             "configs/editing/dress_uv.yml. It merges over configs/default_uv.yml "
                             "automatically -- the UV defaults are a different file, not a "
                             "superset of configs/default.yml.")
    parser.add_argument("--resume", type=int, default=0,
                        help="Any value > 0 resumes the atlas in place from "
                             "<save_dir>/<index>/nuvo_model.ckpt. A gate, not a step number: the "
                             "step always comes from the checkpoint.")
    parser.add_argument("--load_path", type=str, default=None,
                        help="The stage-A PAPR checkpoint (.pth) this atlas is built on. "
                             "Overrides the config's load_path. Required, one way or the other.")
    parser.add_argument("--nuvo_ckpt", type=str, default=None,
                        help="Seed the atlas from an existing nuvo .ckpt. Skips the chart-"
                             "assignment warmup, since the assignment is already learned.")
    parser.add_argument("--bbox_mesh", type=str, default=None,
                        help="A .ply bounding box used to select surface points in 3D instead of "
                             "by the alpha mask. Overrides nuvo.sample.bounding_box_mesh_path.")
    return parser.parse_args()


def dataset_node(values: ConfigNode) -> ConfigNode:
    """The ``dataset`` config node for the training split, with the background sphere stamped on.

    ``get_dataset`` refuses to load ground-truth depth without ``bkg_sphere_radius`` /
    ``bkg_sphere_center``, because the depth loader replaces background pixels with their ray's exit
    point on that sphere and it must be the same sphere the renderer attends to. ``ConfigNode`` is
    read-only, so this is a copy with two extra keys. The stage-A launcher has the same helper --
    duplicated rather than imported so the two entry points do not depend on each other.
    """
    base = values.dataset.to_dict()
    point_opt = values.geoms.points
    base["bkg_sphere_radius"] = point_opt.get("bkg_sphere_radius", 5.0)
    base["bkg_sphere_center"] = point_opt.get("bkg_sphere_center", [0.0, 0.0, 0.0])
    return to_config_node(base, "dataset")


def clear_grad(model: Any) -> None:
    """Zero the stage-A model's gradients through its optimizers."""
    for optimizer in model.optimizers.values():
        if optimizer is not None:
            optimizer.zero_grad()


def train_step(step: int, model: Any, nuvo_model: Nuvo, data: SurfacePointSet, args: ConfigNode,
               device: torch.device) -> float:
    """One atlas optimisation step over a random batch of surface points.

    The batch is drawn with ``randperm`` over the *whole* sample -- typically a million points from
    a hundred views -- so a step sees a spatially unbiased slice of the object rather than one
    view's worth of it. That matters for the cluster and chamfer terms, both of which are computed
    per batch and would otherwise measure a single viewpoint's coverage.

    ``requires_grad`` on the batch is not incidental: the Jacobian distortion loss differentiates
    the UV map with respect to its input, so the input has to be part of the graph.

    Args:
        step: Iteration.
        model: The stage-A model. Frozen in practice; supplies the ``GradScaler`` and is stepped.
        nuvo_model: The atlas.
        data: The surface-point sample from :func:`uv.sampling.extract_surface_points`.
        args: The resolved config tree.
        device: CUDA device.

    Returns:
        The scalar loss. The per-term breakdown is printed by
        :func:`uv.losses.geometry_losses` itself every 200 steps; this release does not log to
        wandb.
    """
    surface_points = data.surface_points
    indices = torch.randperm(surface_points.shape[0])[:args.nuvo.train.G_num]
    batch = surface_points[indices]
    batch.requires_grad = True

    nuvo_model.zero_grad()
    clear_grad(model)

    with torch.autocast(device_type="cuda", dtype=model.amp_dtype, enabled=model.use_amp):
        uv_range = torch.tensor(0.0, device=device)
        geom_losses: dict[str, Tensor] = {"loss_combined": torch.tensor(0.0, device=device)}

        geom_loss_weight = args.texture.uniform_geom_loss_weight
        if args.texture.pcd_geom_loss_weight > 0:
            geom_loss_weight = geom_loss_weight + args.texture.pcd_geom_loss_weight

        if geom_loss_weight > 0:
            charts = nuvo_model.get_all_charts(batch)
            geom_losses = geometry_losses(nuvo_model, step, batch, charts)
            if args.texture.uv_range_loss_weight > 0:
                for pred_uv in charts.pred_uvs:
                    uv_range = uv_range + uv_range_loss(pred_uv)

        geom_loss = geom_losses["loss_combined"]
        loss = geom_loss * geom_loss_weight + uv_range * args.texture.uv_range_loss_weight

        if step % 200 == 0:
            print(" --loss:", loss.item(), "uniform_geom_loss:", geom_loss.item(),
                  "uv_range_loss:", uv_range.item())

        model.scaler.scale(loss).backward()

        model_stepped = optimizer_step(model, step)
        nuvo_stepped = nuvo_model.step(model.scaler, step)
        if model_stepped or nuvo_stepped:
            if args.scaler_min_scale > 0 and model.scaler.get_scale() < args.scaler_min_scale:
                model.scaler.update(args.scaler_min_scale)
            else:
                model.scaler.update()

    return float(loss.item())


def save_nuvo_checkpoint(nuvo_model: Nuvo, step: int, path: Path) -> None:
    """Write the atlas, its optimizers and its schedulers in the archived container format."""
    ckpt: dict[str, Any] = {"step": step, "model_state_dict": nuvo_model.state_dict()}
    for name, optimizer in nuvo_model.optimizers.items():
        ckpt[f"{name}_optimizer"] = optimizer.state_dict()
    for name, scheduler in nuvo_model.schedulers.items():
        ckpt[f"{name}_scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)


def train_atlas(start_step: int, model: Any, nuvo_model: Nuvo, data: SurfacePointSet,
                args: ConfigNode, device: torch.device, run_dir: Path) -> None:
    """Stage 1's loop: fit the chart networks to the fixed surface-point cloud."""
    iters = args.nuvo.train.iters
    print("Start step:", start_step, "Total steps:", iters)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    history: dict[str, list[float]] = {"steps": [], "train_losses": []}
    avg_train_loss = 0.0
    eval_step_cnt = 0
    start_time = time.time()

    for step in range(start_step, iters):
        loss = train_step(step, model, nuvo_model, data, args, device)
        avg_train_loss += loss
        eval_step_cnt += 1

        if step % 200 == 0:
            print("Train step:", step, "loss:", loss, "scale:", model.scaler.get_scale(),
                  f"time: {time.time() - start_time:.2f}s")
            start_time = time.time()

        next_step = step + 1
        if (next_step % 5000 == 0 or (next_step % 2000 == 0 and next_step < 20000)
                or (next_step % 100 == 0 and next_step < 1000) or next_step == 1):
            history["steps"].append(next_step)
            history["train_losses"].append(avg_train_loss / eval_step_cnt)
            print("Eval step:", next_step, "train_loss:", history["train_losses"][-1])
            avg_train_loss = 0.0
            eval_step_cnt = 0

            save_nuvo_checkpoint(nuvo_model, next_step, run_dir / "nuvo_model.ckpt")
            if next_step % 10000 == 0 or (next_step % 5000 == 0 and next_step < 20000):
                save_nuvo_checkpoint(nuvo_model, next_step,
                                     run_dir / f"nuvo_model_{next_step}.ckpt")
            torch.save({k: torch.tensor(v) for k, v in history.items()},
                       run_dir / "loss_and_lr.pth")

    save_nuvo_checkpoint(nuvo_model, iters, run_dir / "nuvo_model.ckpt")
    print("Mapping MLP training finished!")


def export_texture_maps(nuvo_model: Nuvo, out_dir: Path) -> None:
    """Write the atlas as per-chart PNGs, one packed grid, and the raw tensor.

    The packed grid is the editable artifact: open it, paint on it, and
    ``render_edit.py --texture_is_grid`` reads it back into the same chart order. Its layout is
    :func:`uv.texture.grid_layout` of this atlas's chart count and nothing else -- the four-chart
    dress exports as one row of four 256px charts, a 1024x256 strip, which is what the archived
    exports and the shipped demo grid are.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        texture_map = nuvo_model.get_rgb_for_texture_map()
    charts = texture_map.detach().cpu().numpy()

    for chart_idx in range(charts.shape[0]):
        chart = np.clip(charts[chart_idx][..., :3], 0, 1)
        imageio.imwrite(out_dir / f"texture_chart_{chart_idx:02d}.png",
                        (chart * 255).astype(np.uint8))
    rows, columns = grid_layout(charts.shape[0])
    imageio.imwrite(out_dir / "texture_map_grid.png",
                    pack_texture_map_to_grid(charts[..., :3]))
    torch.save(texture_map.cpu(), out_dir / "texture_map.pt")
    print(f"  Saved {charts.shape[0]} chart textures to: {out_dir}")
    print(f"  Packed grid: texture_map_grid.png ({rows} row(s) x {columns} column(s))")


def train_texture_map(nuvo_model: Nuvo, data: SurfacePointSet, args: ConfigNode,
                      device: torch.device, run_dir: Path) -> None:
    """Stage 2: fit the per-chart texture images to the observed colours.

    Every chart network is frozen first -- not just excluded from the optimizer, but
    ``requires_grad = False`` -- so this is a pure image fit through a fixed lookup. Combined with
    ``texture_map_stop_uv_grad`` / ``texture_map_stop_prob_grad``, that means the texture cannot
    reshape the atlas to make itself easier to fit.

    Only the first ``num_original_surface_points`` rows are used: the appended point cloud has
    positions but no observed colour, and fitting it against another point's colour would smear the
    texture.

    Args:
        nuvo_model: The trained atlas.
        data: The same sample stage 1 used.
        args: The resolved config tree.
        device: CUDA device.
        run_dir: Where the checkpoints and the exported maps go.
    """
    print("=" * 60)
    print("Starting Texture Map Training")
    print("=" * 60)

    train_conf = args.nuvo.train
    optimizer_conf = args.nuvo.optimizer
    texture_train_iters = train_conf.texture_train_iters
    batch_size = train_conf.G_num
    log_interval = 200
    save_interval = train_conf.texture_map_save_interval

    print(f"Texture training iterations: {texture_train_iters}")
    print(f"Texture map learning rate: {optimizer_conf.texture_map_lr}")
    print(f"Batch size: {batch_size}")

    texture_log_dir = run_dir / "texture_training"
    texture_log_dir.mkdir(parents=True, exist_ok=True)

    print("\nFreezing mapping MLPs...")
    frozen = (nuvo_model.chart_assignment_mlp, nuvo_model.texture_coordinate_mlp,
              nuvo_model.surface_coordinate_mlp)
    for module in frozen:
        for param in module.parameters():
            param.requires_grad = False
    nuvo_model.sigma.requires_grad = False

    texture_params: list[dict[str, Any]] = []
    if nuvo_model.texture_map is not None:
        texture_params.append({"params": [nuvo_model.texture_map],
                               "lr": optimizer_conf.texture_map_lr})
        print(f"  - texture_map: trainable (lr={optimizer_conf.texture_map_lr})")
    if nuvo_model.texture_map_bkg_feat is not None:
        texture_params.append({"params": [nuvo_model.texture_map_bkg_feat],
                               "lr": optimizer_conf.texture_map_bkg_feat_lr})
        print(f"  - texture_map_bkg_feat: trainable "
              f"(lr={optimizer_conf.texture_map_bkg_feat_lr})")
    if nuvo_model.texture_mlp is not None:
        texture_params.append({"params": list(nuvo_model.texture_mlp.parameters()),
                               "lr": optimizer_conf.texture_mlp_lr})
    if not texture_params:
        print("No texture parameters to train! Skipping texture training.")
        return

    optimizer = torch.optim.Adam(texture_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=texture_train_iters)

    num_points = data.num_original_surface_points
    surface_points = data.surface_points[:num_points]
    target_rgb = data.rgb
    print(f"\nTotal surface points for texture training: {num_points} "
          f"(original surface points only)")
    print(f"Target RGB shape: {tuple(target_rgb.shape)}")

    loss_fn = nn.MSELoss()
    train_losses: list[float] = []
    start_time = time.time()

    print(f"\nStarting texture training for {texture_train_iters} iterations...")
    for step in tqdm(range(texture_train_iters), desc="Texture Training"):
        batch_indices = torch.randperm(num_points, device=device)[:batch_size]

        optimizer.zero_grad()
        dummy = torch.zeros((batch_size, 3), device=device)
        pred_rgb, _, _ = nuvo_model.get_rgb_from_texture_map_for_points(
            surface_points[batch_indices], dummy, dummy, normals=None,
            argmax=args.nuvo.texture.train_argmax, render_checkboard=False,
            pred_uvs=None, pred_chart_probs=None, texture_map=None,
            return_additional_texture_map=False,
        )
        loss = loss_fn(pred_rgb, target_rgb[batch_indices])
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(float(loss.item()))

        if step % log_interval == 0 or step == texture_train_iters - 1:
            window = train_losses[-log_interval:] if len(train_losses) >= log_interval \
                else train_losses
            print(f"  Step {step}: loss={loss.item():.6f}, avg_loss={np.mean(window):.6f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}, time={time.time() - start_time:.1f}s")

        if step % save_interval == 0 or step == texture_train_iters - 1:
            torch.save({"step": step, "model_state_dict": nuvo_model.state_dict(),
                        "texture_optimizer": optimizer.state_dict(),
                        "texture_scheduler": scheduler.state_dict(),
                        "train_losses": train_losses},
                       texture_log_dir / f"texture_checkpoint_{step:06d}.ckpt")

    torch.save({"step": texture_train_iters, "model_state_dict": nuvo_model.state_dict(),
                "texture_optimizer": optimizer.state_dict(),
                "texture_scheduler": scheduler.state_dict(), "train_losses": train_losses},
               texture_log_dir / "texture_final.ckpt")
    torch.save({"step": texture_train_iters, "model_state_dict": nuvo_model.state_dict()},
               run_dir / "nuvo_model_with_texture.ckpt")

    print("\nUnfreezing mapping MLPs...")
    for module in frozen:
        for param in module.parameters():
            param.requires_grad = True
    nuvo_model.sigma.requires_grad = True

    print("\nSaving learned texture map...")
    export_texture_maps(nuvo_model, run_dir / "learned_texture_map")

    print("=" * 60)
    print("Texture Map Training Complete!")
    print(f"Final loss: {train_losses[-1]:.6f}")
    print(f"Logs saved to: {texture_log_dir}")
    print("=" * 60)


def load_stage_a(model: Any, load_path: str) -> None:
    """Load the trained PAPR checkpoint this atlas is built on, and build its optimizers.

    The optimizers come *after* the load, because the checkpoint's point count is almost never the
    config's and an optimizer holding the pre-load parameter objects would update tensors the model
    no longer has. They exist only so the training step can step and zero them -- no stage-A
    parameter receives a gradient in this stage.
    """
    path = Path(load_path) if ".pth" in load_path else Path(load_path) / "model.pth"
    report = model.load(path)
    print("!!!!! Loaded model from %s at step %s (%d points)"
          % (path, report.step, report.n_points))
    init_optimizers(model, 0)


def load_nuvo_checkpoint(nuvo_model: Nuvo, path: str | Path, device: torch.device) -> int:
    """Copy an archived atlas into ``nuvo_model``; returns the checkpoint's step.

    Accepts all three archived container shapes: a dict with ``model_state_dict``, a
    single-entry ``{step: state_dict}``, or a bare state dict. Loading is non-strict via
    :meth:`uv.nuvo.Nuvo.load_model`, so seeding a differently-sized atlas keeps whatever fits.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nuvo model checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif len(checkpoint) == 1 and isinstance(list(checkpoint.values())[0], dict):
        state_dict = list(checkpoint.values())[0]
    else:
        state_dict = checkpoint
    nuvo_model.load_model(state_dict)
    step = checkpoint.get("step", 0) if isinstance(checkpoint, dict) else 0
    print(f"Loaded Nuvo model checkpoint from: {path} (Step: {step})")
    return int(step)


def restore_nuvo_optimizers(nuvo_model: Nuvo, path: Path, device: torch.device) -> None:
    """Restore optimizer and scheduler state from a ``nuvo_model.ckpt``, for ``--resume``.

    Adam's moments matter here in a way they do not in stage A's resume: the texture map's learning
    rate is 0.02 against 1e-4 for the chart networks, and restarting its moment estimates puts a
    large uncorrected first step into an atlas that had converged.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    for name, optimizer in nuvo_model.optimizers.items():
        state = checkpoint.get(f"{name}_optimizer")
        if state is not None:
            optimizer.load_state_dict(state)
    for name, scheduler in nuvo_model.schedulers.items():
        state = checkpoint.get(f"{name}_scheduler")
        if state is not None:
            scheduler.load_state_dict(state)


def main() -> None:
    cli = parse_args()
    config = load_config(cli.opt)
    values = config.values

    run_dir = make_run_dir(config)
    sys.stdout = Logger(run_dir / "train.log", sys.stdout)
    sys.stderr = Logger(run_dir / "train_error.log", sys.stderr)
    archive_config(config, run_dir)
    snapshot_code(run_dir / "code.zip")
    print("Config:", config.source, "sha256:", config.config_sha256)
    print("Fingerprint:", config.fingerprint, "-> run dir:", run_dir)

    setup_seed(values.seed)
    device = setup_torch(config)

    model = get_model(values, device)
    dataset = get_dataset(dataset_node(values), mode="train", device=device)
    model = model.to(device)
    model.set_camera(dataset.focal_x, dataset.focal_y, dataset.cx, dataset.cy,
                     dataset.H, dataset.W)

    load_path = cli.load_path if cli.load_path is not None else values.load_path
    if not load_path:
        raise ValueError(
            "stage b needs a trained PAPR checkpoint: pass --load_path <model.pth>, or set "
            "load_path in the config. Learning an atlas over a randomly initialised point cloud "
            "parameterises noise."
        )
    load_stage_a(model, str(load_path))

    nuvo_model = Nuvo(values.nuvo, device=device).to(device)
    start_step = 0
    seeded_from_checkpoint = False
    if cli.resume > 0:
        resume_path = run_dir / "nuvo_model.ckpt"
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"--resume was given but there is no atlas checkpoint at {resume_path}. A resume "
                f"continues a run in place, in <save_dir>/<index>; to start FROM an atlas "
                f"elsewhere, use --nuvo_ckpt."
            )
        start_step = load_nuvo_checkpoint(nuvo_model, resume_path, device)
        seeded_from_checkpoint = True
        print("!!!!! Resuming atlas from step %s" % start_step)
    elif cli.nuvo_ckpt:
        load_nuvo_checkpoint(nuvo_model, cli.nuvo_ckpt, device)
        seeded_from_checkpoint = True

    nuvo_model.init_optimizers(values.training.steps)
    if cli.resume > 0:
        restore_nuvo_optimizers(nuvo_model, run_dir / "nuvo_model.ckpt", device)
    nuvo_model.fix_groups(values.nuvo.train.fix_keys)

    bbox_path = cli.bbox_mesh or values.nuvo.sample.bounding_box_mesh_path
    bbox_hull = None
    if bbox_path:
        bbox_hull = load_bounding_box(bbox_path, coord_scale=1.0)
        print("Using bounding box for surface point filtering (instead of fg_mask)")

    data = extract_surface_points(
        model, dataset, values, bbox_hull=bbox_hull,
        use_gt_surface_points=values.nuvo.sample.use_gt_surface_points,
    )

    preview = data.surface_points[:data.num_original_surface_points]
    colors = data.rgb
    if preview.shape[0] > 300000:
        keep = torch.randperm(preview.shape[0])[:300000]
        preview, colors = preview[keep], colors[keep]
    write_ply_points(run_dir / "sampled_surface_points.ply", preview.cpu().numpy(),
                     colors=colors.cpu().numpy())
    print("Saved sampled surface points to", run_dir / "sampled_surface_points.ply")

    if seeded_from_checkpoint:
        print("Skipping chart assignment warmup because the assignment is loaded from a "
              "checkpoint.")
    else:
        warmup_points = model.points.detach()
        if values.texture.divide_by_coord_scale:
            warmup_points = warmup_points / model.coord_scale
        warmup_chart_assignment(nuvo_model, values.nuvo, warmup_points, log_dir=run_dir)

    train_atlas(start_step, model, nuvo_model, data, values, device, run_dir)
    if torch.cuda.is_available():
        print(torch.cuda.memory_summary())

    print("\n" + "=" * 60)
    print("Stage 2: Learning Texture Map")
    print("=" * 60 + "\n")
    train_texture_map(nuvo_model, data, values, device, run_dir)


if __name__ == "__main__":
    main()

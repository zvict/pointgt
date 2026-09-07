from __future__ import annotations

import argparse
import bisect
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import imageio.v2 as imageio
import numpy as np
import torch
from torch import Tensor

from dataset import get_dataset, get_loader
from models import get_model
from models.checkpoint import extra_path, save_checkpoint
from models.densify import add_points, get_prune_signal, prune_points
from models.losses import build_loss
from models.optim import init_optimizers
from models.optim import step as optimizer_step
from models.regularizers import compute_regularizers
from utils.config import deep_merge, depth_root, load_config
from utils.ply import save_points
from utils.runtime import ConfigNode, to_config_node
from utils.session import (
    Logger,
    archive_config,
    make_run_dir,
    setup_seed,
    setup_torch,
    snapshot_code,
)

#: How many recent iteration times the throughput line averages over.
_ITER_WINDOW = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PointGT stage-a training")
    parser.add_argument("--opt", type=str, required=True,
                        help="Path to a scene config (.yml), e.g. configs/nerfsyn/lego.yml")
    parser.add_argument("--resume", type=int, default=0,
                        help="Any value > 0 resumes in place from <save_dir>/<index>/model.pth. "
                             "This is a gate, not a step number: the step ALWAYS comes from the "
                             "checkpoint, so --resume 1 and --resume 150000 do the same thing.")
    parser.add_argument("--load_path", type=str, default=None,
                        help="A .pth to fine-tune FROM (or a directory holding model.pth). The "
                             "step counter restarts at 0 and training.steps is the absolute cap. "
                             "Overrides the config's own load_path.")
    return parser.parse_args()


def dataset_node(values: ConfigNode, *, split: str) -> ConfigNode:
    """The ``dataset`` config node for one split, with the background sphere stamped on.

    ``geoms.points.bkg_sphere_radius`` / ``bkg_sphere_center`` are copied into the dataset node
    because the depth loader substitutes background pixels with their ray's exit point on that
    sphere, and it must be the *same* sphere the renderer attends to. ``ConfigNode`` is read-only,
    so this returns a copy with two extra keys stamped on, rather than an in-place mutation.

    Args:
        values: The resolved config tree.
        split: ``"train"`` for ``dataset``, ``"eval"`` for ``eval.dataset`` merged over it.
    """
    base = values.dataset.to_dict()
    if split == "eval":
        base = deep_merge(base, values.eval.dataset.to_dict())
    elif split != "train":
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'eval'")

    point_opt = values.geoms.points
    base["bkg_sphere_radius"] = point_opt.get("bkg_sphere_radius", 5.0)
    base["bkg_sphere_center"] = point_opt.get("bkg_sphere_center", [0.0, 0.0, 0.0])
    return to_config_node(base, "dataset")


def _depth_help(node: ConfigNode, error: Exception) -> str:
    """The message a missing 2DGS depth directory earns.

    The dataset *raises* instead of zero-filling. Zero depth back-projects onto the
    camera centre, so the surface-point loss would quietly pull the cloud into the camera -- the
    run would finish and the checkpoint would be wrong. This adds the two commands that produce
    the maps and the paths this process actually looked at.

    2DGS is deliberately not launched for you: it lives in a different conda environment under a
    research-only licence, and spending several GPU-minutes inside a foreign environment is not a
    thing a training script should decide to do.

    Only the *first line* of the dataset's own message is quoted -- the frame it failed on. The
    rest of it is already on screen: this is raised ``from error``, so Python prints the original
    exception in full immediately above.
    """
    depth_dir = Path(str(node.gt_depth_dir))
    model_dir = depth_dir
    for _ in range(3):
        if model_dir.parent != model_dir:
            model_dir = model_dir.parent
    return (
        f"{str(error).splitlines()[0]}\n"
        f"\n"
        f"Resolved from this config:\n"
        f"    dataset.path         {node.path}\n"
        f"    dataset.gt_depth_dir {depth_dir}\n"
        f"    2dgs/ resolves to    {depth_root()}   (override with POINTGT_2DGS_ROOT)\n"
        f"\n"
        f"Produce the maps with 2D Gaussian Splatting, in ITS OWN conda environment "
        f"(https://github.com/hbb1/2d-gaussian-splatting):\n"
        f"    python train.py  -s {node.path} -m {model_dir} \\\n"
        f"        --eval --white_background --test_iterations -1 --quiet\n"
        f"    python render.py -s {node.path} -m {model_dir} --iteration 30000 \\\n"
        f"        --eval --white_background --skip_train --skip_mesh --quiet\n"
        f"\n"
        f"They are not run automatically: 2DGS is a separate environment under a research-only "
        f"licence. Set dataset.load_gt_depth: false to train without depth supervision -- but "
        f"note that the editing configs weight the surface-point loss at 0.1, so turning it off "
        f"there optimises a different objective than the released checkpoints."
    )


def build_datasets(values: ConfigNode, device: torch.device) -> tuple[Any, Any]:
    """The training split and the single-frame validation split.

    The eval split never loads ground-truth depth. ``get_full_img`` structurally
    never emits ``surface_points``, so the surface-point loss is skipped in every validation render
    regardless.
    """
    train_node = dataset_node(values, split="train")
    eval_node = dataset_node(values, split="eval")
    try:
        train_dataset = get_dataset(train_node, mode="train", device=device)
    except FileNotFoundError as error:
        raise FileNotFoundError(_depth_help(train_node, error)) from error
    eval_dataset = get_dataset(eval_node, mode=eval_node.get("mode", "test"), device=device)
    return train_dataset, eval_dataset


@dataclass
class TrainingHistory:
    """The curves ``loss_and_lr.pth`` carries, one entry per eval.

    Its own object because ``--resume`` has to restore it and the eval step has to append to it.
    The file format is unchanged, so an archived ``loss_and_lr.pth`` still loads.
    """

    steps: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    eval_psnrs: list[float] = field(default_factory=list)
    pt_lrs: list[float] = field(default_factory=list)
    attn_lrs: list[float] = field(default_factory=list)

    #: Name in the ``.pth`` -> attribute.
    FIELDS = ("steps", "train_losses", "eval_losses", "eval_psnrs", "pt_lrs", "attn_lrs")

    def save(self, path: str | Path) -> None:
        torch.save({name: torch.tensor(getattr(self, name)) for name in self.FIELDS}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrainingHistory":
        payload = torch.load(path, weights_only=True)
        return cls(**{name: payload[name].tolist() for name in cls.FIELDS})


class TrainStepOutput(NamedTuple):
    """What one step produced, for the log line and the eval figure."""

    loss: float
    rgb: Tensor
    target: Tensor
    log: dict[str, float]


def clear_grad(model: Any) -> None:
    """Zero every optimizer's gradients.

    Iterating the *optimizers* rather than ``model.parameters()`` is deliberate: a parameter that
    ``training.fix_keys`` froze has no optimizer, and its stale gradient is left alone.
    """
    for optimizer in model.optimizers.values():
        if optimizer is not None:
            optimizer.zero_grad()


def _background_color(model: Any, args: ConfigNode, target: Tensor, mask: Tensor, step: int,
                      device: torch.device) -> Tensor:
    """The background colour this step composites against.

    With ``rnd_background`` (false in all nine shipped configs) the background is re-randomised
    every step and the *target's* background pixels are overwritten to match, which is what teaches
    the model to separate foreground from background rather than to memorise white. ``target`` is
    mutated in place -- it is a fresh collate output, not a view of the dataset.
    """
    if not args.rnd_background:
        return model.bkg_feats * model.bkg_scaler.squeeze(0)

    start, stop = args.rnd_background_start, args.rnd_background_stop
    ramp = (step - start) / (stop - start) if (stop - start > 0 and step >= start) else 0.0
    shift = ramp + args.rnd_background_shift
    if step % 200 == 0:
        print(f"**** rnd background shift: {shift} start: {start} stop: {stop}")
    rnd_bkg_color = torch.rand(3).to(device) + shift
    rnd_bkg_color = rnd_bkg_color * model.bkg_scaler.squeeze(0)
    target[mask.squeeze(-1) < 0.5] = rnd_bkg_color
    return rnd_bkg_color[None, :]


def train_forward_backward(step: int, model: Any, device: torch.device, dataset: Any,
                           batch: dict, loss_fn: Any, args: ConfigNode) -> TrainStepOutput:
    """Render one batch of patches, build the loss, and back-propagate it.

    Gradients are zeroed at the top of the step rather than after the optimizer step: it is the
    last moment at which zeroing is unambiguously correct, since the densification surgery between
    the backward and the next forward replaces whole parameters.
    """
    target = batch["image"].to(device)
    mask = batch["mask"].to(device)
    rays_d = batch["rayd"].to(device)
    rays_o = batch["rayo"].to(device)
    pix_coords = batch["pix_coords"].to(device)
    c2w = dataset.get_c2w(batch["idx"]).to(device)
    gt_surface_points = None
    if "surface_points" in batch:
        gt_surface_points = batch["surface_points"].to(device)

    bkg_color = _background_color(model, args, target, mask, step, device)
    clear_grad(model)

    with torch.autocast(device_type="cuda", dtype=model.amp_dtype, enabled=model.use_amp):
        out = model(rays_o, rays_d, c2w, pix_coords, bkg_color, step)
        rgb = model.last_act(out.rgb)
        rgb_loss = loss_fn(rgb, target)
        loss = rgb_loss

        regularizers = compute_regularizers(
            args,
            step=step,
            device=device,
            points=out.cloud.points,
            rays_o=rays_o,
            rays_d=rays_d,
            select_k_ind=out.select_k_ind,
            attn=out.attn,
            bkg_attn=out.bkg_attn,
            mask=mask,
            sphere_intersection=out.sphere_intersection,
            coord_scale=model.coord_scale,
            points_influ_scores=model.points_influ_scores,
            gt_points=model.gt_points,
            gt_surface_points=gt_surface_points,
            pruned_points=model.pruned_points,
        )

        log: dict[str, float] = {"losses/total": 0.0, "losses/rgb": float(rgb_loss.item())}
        for name, (value, weight) in regularizers.items():
            if weight <= 0:
                continue
            if args.training.get(f"{name}_scale_by_rgb_loss", False) \
                    and value.item() > rgb_loss.item():
                value = value * (rgb_loss.item() / value.item())
            loss = loss + value * weight
            log[f"losses/regularizer_{name}"] = float(value.item())
            log[f"losses/regularizer_{name}_weight"] = float(weight)
            if step % 200 == 0:
                print(f" --{name}:", value.item(), "weight:", weight,
                      "weighted loss:", value.item() * weight)

    model.scaler.scale(loss).backward()
    log["losses/total"] = float(loss.item())

    if step % 200 == 0:
        scalers = [f"{x:.4f}" for x in model.bkg_scaler.squeeze(0).tolist()]
        print(" --rgb loss:", rgb_loss.item(), "--total loss:", loss.item(),
              "--bkg scaler:", scalers)

    return TrainStepOutput(float(loss.item()), rgb.detach(), target, log)


def prune_threshold(args: ConfigNode, model: Any, step: int) -> tuple[float, bool]:
    """This step's prune threshold, and whether pruning should be skipped entirely.

    Four ways to arrive at a threshold, checked in this precedence order. Only the first --
    ``training.prune_thresh``, a constant -- is used by any shipped config; the other three are
    kept because they are documented config keys and :func:`models.densify.prune_points` takes the
    threshold from its caller precisely so that the policy can live here.

    Returns:
        ``(threshold, skip)``. ``skip`` is only ever true on the quantile path, which refuses to
        prune a cloud that has not yet reached ``max_num_pts``: a quantile always nominates a fixed
        *fraction* of the cloud, so applying it to a still-growing cloud deletes points the
        addition schedule just paid for.
    """
    training = args.training
    thresh = training.prune_thresh
    quantile_range = (training.prune_thresh_quantile_start is not None
                      and training.prune_thresh_quantile_end is not None)

    if len(training.prune_steps_list) > 0:
        index = bisect.bisect_left(training.prune_steps_list, step)
        return training.prune_thresh_list[index], False

    if training.prune_thresh_quantile is not None or quantile_range:
        if args.max_num_pts > 0 and model.points.shape[0] < args.max_num_pts * 0.95:
            return thresh, True
        if quantile_range:
            quantile = _interpolate(training.prune_thresh_quantile_start,
                                    training.prune_thresh_quantile_end, training, step)
        else:
            quantile = training.prune_thresh_quantile
        signal = get_prune_signal(model).squeeze()
        return torch.quantile(signal, quantile).item(), False

    if training.prune_thresh_start is not None and training.prune_thresh_end is not None:
        return _interpolate(training.prune_thresh_start, training.prune_thresh_end,
                            training, step), False

    return thresh, False


def _interpolate(start_value: float, end_value: float, training: ConfigNode, step: int) -> float:
    """Linear ramp across the pruning window, clamped at both ends."""
    span = training.prune_stop - training.prune_start
    if span <= 0:
        return start_value
    progress = max(0.0, min(1.0, (step - training.prune_start) / span))
    return start_value + progress * (end_value - start_value)


def densify(step: int, model: Any, args: ConfigNode) -> float:
    """Prune, then add. Returns the prune threshold, which the add path also reads.

    The order is not cosmetic: ``add_points`` samples where the *surviving* points have large
    accumulated gradient, so adding before pruning would clone points that are about to be deleted.
    ``model.pruned_points`` is what enforces it, and it flips true at the first prune whether or
    not that prune actually removed anything.
    """
    training = args.training
    thresh = training.prune_thresh
    skip_prune = False

    if training.prune_steps > 0 and training.prune_start <= step < training.prune_stop:
        thresh, skip_prune = prune_threshold(args, model, step)
        if step % training.prune_steps == 0 and not skip_prune:
            num_pruned = prune_points(model, thresh, step=step)
            model.pruned_points = True
            print("Step %d: Pruned %d points, prune threshold %f" % (step, num_pruned, thresh))

    prune_disabled = training.prune_start >= training.steps
    ready = model.pruned_points or prune_disabled

    if ready and len(training.add_steps_list) > 0:
        if step in training.add_steps_list:
            add_num = training.add_num_list[training.add_steps_list.index(step)]
            _add(model, args, add_num, step, thresh)
    elif ready and training.add_steps > 0 and step % training.add_steps == 0 \
            and training.add_start <= step < training.add_stop:
        _add(model, args, training.add_num, step, thresh)

    return thresh


def _add(model: Any, args: ConfigNode, add_num: int, step: int, prune_thresh: float) -> None:
    """Add ``add_num`` points, clamped so the cloud cannot overshoot ``max_num_pts``."""
    if args.max_num_pts > 0:
        add_num = min(add_num, args.max_num_pts - model.points.shape[0])
    if add_num <= 0:
        return
    num_added = add_points(model, add_num, step=step, prune_thresh=prune_thresh)
    model.added_points = True
    print("Step %d: Added %d points" % (step, num_added))


def train_update(step: int, model: Any, args: ConfigNode) -> None:
    """The optimizer half of a step: densification, the guard stack, the steps, the scaler.

    Everything here is outside the autograd graph -- the gradients are already computed -- so the
    whole block runs under ``no_grad``, which is also what makes the in-place parameter surgery in
    :mod:`models.densify` legal.
    """
    with torch.no_grad():
        densify(step, model, args)

        optimizer_step(
            model, step,
            grad_clip_norm=args.training.get("grad_clip_norm", None),
            grad_clip_value=args.training.get("grad_clip_value", None),
        )

        if step % 200 == 0:
            grad = model.points.grad
            print("max points grad:", grad.max().item() if grad is not None else "None (frozen)")

        if args.scaler_min_scale > 0 and model.scaler.get_scale() < args.scaler_min_scale:
            model.scaler.update(args.scaler_min_scale)
        else:
            model.scaler.update()


def train_step(step: int, model: Any, device: torch.device, dataset: Any, batch: dict,
               loss_fn: Any, args: ConfigNode) -> TrainStepOutput:
    """One full training step: forward, backward, densify, optimize."""
    out = train_forward_backward(step, model, device, dataset, batch, loss_fn, args)
    train_update(step, model, args)
    return out


@torch.no_grad()
def render_frame(model: Any, data: dict, c2w: Tensor, args: ConfigNode, step: int) -> Tensor:
    """Render one full frame in tiles and decode it once. Returns ``(N, H, W, 3)`` in ``[0, 1]``.

    The attention over a whole 800x800 frame does not fit in memory, so the frame is rendered in
    ``eval.max_height`` x ``eval.max_width`` tiles -- but the tiles are assembled into a single
    feature frame and the decoder runs **once** over all of it. Decoding per tile would put a U-Net
    receptive-field boundary at every seam.
    """
    rays_o = data["rayo"]
    rays_d = data["rayd"]
    pix_coords = data["pix_coords"]
    n, height, width, _ = rays_d.shape

    tile_h = min(args.eval.max_height, height) if args.eval.max_height > 0 else height
    tile_w = min(args.eval.max_width, width) if args.eval.max_width > 0 else width

    feature_map: Tensor | None = None
    for top in range(0, height, tile_h):
        for left in range(0, width, tile_w):
            bottom, right = min(top + tile_h, height), min(left + tile_w, width)
            out = model.evaluate(
                rays_o,
                rays_d[:, top:bottom, left:right],
                c2w,
                pix_coords[top:bottom, left:right],
                step=step,
            )
            if feature_map is None:
                feature_map = torch.empty(n, height, width, 1, out.fused_features.shape[-1],
                                          device=rays_d.device)
            feature_map[:, top:bottom, left:right] = out.fused_features

    rgb, _ = model.decode(feature_map.squeeze(-2), n, height, width)
    return torch.clamp(model.last_act(rgb), 0, 1)


def save_comparison(path: str | Path, pred: Tensor, target: Tensor) -> None:
    """Write ``pred | ground truth`` side by side as an 8-bit PNG."""
    frames = [t.squeeze(0).detach().cpu().numpy() for t in (pred, target)]
    combined = np.concatenate([np.clip(f, 0.0, 1.0) for f in frames], axis=1)
    imageio.imwrite(str(path), (combined * 255).astype(np.uint8))


def save_run_checkpoint(model: Any, step: int, run_dir: Path) -> None:
    """``model.pth`` plus the optimizer/scheduler/scaler sidecars.

    Only the scaler is ever read back (``--resume`` rebuilds optimizers and replays schedulers --
    see :mod:`models.optim`), but all four are written because an external tool may expect all
    four sidecars to exist.
    """
    optimizers = {name: (opt.state_dict() if opt is not None else None)
                  for name, opt in model.optimizers.items()}
    schedulers = {name: (sch.state_dict() if sch is not None else None)
                  for name, sch in model.schedulers.items()}
    model.save(step, run_dir, optimizers=optimizers, schedulers=schedulers)


def eval_step(step: int, model: Any, eval_dataset: Any, loss_fn: Any, args: ConfigNode,
              history: TrainingHistory, run_dir: Path) -> None:
    """Render the held-out frame, log the metrics, and checkpoint.

    This is also the only place a checkpoint is written, which is why the eval cadence is what
    determines how much work a crash costs.
    """
    data = eval_dataset.get_full_img(args.eval.img_idx)
    target = data["image"]
    c2w = eval_dataset.get_c2w([args.eval.img_idx])

    rgb = render_frame(model, data, c2w, args, step)

    eval_loss = float(loss_fn(rgb, target).item())
    mse = float(((rgb - target) ** 2).mean().item())
    eval_psnr = -10.0 * np.log(mse) / np.log(10.0)

    history.eval_losses.append(eval_loss)
    history.eval_psnrs.append(float(eval_psnr))
    print("Eval step:", step, "train_loss:", history.train_losses[-1],
          "eval_loss:", eval_loss, "eval_psnr:", eval_psnr)

    if args.eval.save_fig:
        renders = run_dir / "eval_renders"
        renders.mkdir(parents=True, exist_ok=True)
        save_comparison(renders / ("iter_%d.png" % step), rgb, target)

    save_run_checkpoint(model, step, run_dir)
    points = model.points.detach().cpu().numpy()
    save_points(run_dir / "points_normed.ply", points / model.coord_scale)

    if step % 25000 == 0 or (step % 1000 == 0 and step < 20000):
        save_checkpoint(run_dir / ("model_%d.pth" % step), model, step)
        save_points(run_dir / ("points_%d_normed.ply" % step), points / model.coord_scale)

    history.save(run_dir / "loss_and_lr.pth")


def train_and_eval(start_step: int, model: Any, device: torch.device, dataset: Any,
                   eval_dataset: Any, history: TrainingHistory, args: ConfigNode,
                   run_dir: Path) -> None:
    """The loop. ``step`` counts optimizer updates and is what every schedule is expressed in."""
    loader = get_loader(dataset, args.dataset, mode="train")
    model.set_camera(dataset.focal_x, dataset.focal_y, dataset.cx, dataset.cy,
                     dataset.H, dataset.W)

    loss_fn = build_loss(args.training.losses).to(device)
    print("Loss:", loss_fn)

    save_points(run_dir / "points_init.ply", model.points.detach().cpu().numpy())

    step = start_step
    avg_train_loss = 0.0
    eval_step_cnt = 0
    iter_times: deque[float] = deque(maxlen=_ITER_WINDOW)

    print("Start step:", start_step, "Total steps:", args.training.steps)
    start_time = time.time()
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    while step < args.training.steps:
        for batch in loader:
            iter_start = time.perf_counter()
            out = train_step(step, model, device, dataset, batch, loss_fn, args)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            iter_times.append(time.perf_counter() - iter_start)

            if step % 100 == 0:
                rolling = sum(iter_times) / len(iter_times)
                line = f"[obs] step={step} iter_s_rolling{_ITER_WINDOW}={rolling:.4f}"
                if torch.cuda.is_available():
                    line += f" vram_peak_GB={torch.cuda.max_memory_allocated() / 1024 ** 3:.3f}"
                print(line)
            if step % 200 == 0:
                elapsed = time.time() - start_time
                print("Train step:", step, "loss:", out.loss, "attn_lr:", model.attn_lr,
                      "pts_lr:", model.pts_lr, "scale:", model.scaler.get_scale(),
                      f"time: {elapsed:.2f}s")
                start_time = time.time()

            avg_train_loss += out.loss
            step += 1
            eval_step_cnt += 1

            if (step % args.eval.step == 0) or (step % 500 == 0 and step < 20000) or step == 1:
                history.steps.append(step)
                history.train_losses.append(avg_train_loss / eval_step_cnt)
                history.pt_lrs.append(model.pts_lr)
                history.attn_lrs.append(model.attn_lr)
                eval_step(step, model, eval_dataset, loss_fn, args, history, run_dir)
                avg_train_loss = 0.0
                eval_step_cnt = 0

            if step >= args.training.steps:
                break

    print("Training finished!")


def restore_scaler(model: Any, checkpoint: Path) -> None:
    """Restore the AMP loss scale from a checkpoint's ``scaler.pth`` sidecar, if it has one.

    Only the scaler is restored here (``PAPR.load``'s ``load_optimizer`` defaults to ``False``).
    Restoring it matters: a converged run sits at a scale that took thousands of steps of doubling
    to reach, and starting over at 65536 spends the first steps after a resume overflowing and
    halving.
    """
    path = extra_path(checkpoint, "scaler")
    if path.exists():
        model.scaler.load_state_dict(torch.load(path, weights_only=True))
        print("Restored GradScaler state from %s (scale %s)" % (path, model.scaler.get_scale()))


def resume(model: Any, run_dir: Path) -> tuple[int, TrainingHistory]:
    """Continue a run in place from ``<run_dir>/model.pth``.

    The step comes from the checkpoint, never from ``--resume``'s value.
    """
    checkpoint = run_dir / "model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"--resume was given but there is no checkpoint at {checkpoint}. A resume continues a "
            f"run in place, in <save_dir>/<index>; to start FROM a checkpoint elsewhere, and from "
            f"step 0, use --load_path."
        )
    report = model.load(checkpoint)
    print("!!!!! Resuming from step %s (%d points)" % (report.step, report.n_points))

    init_optimizers(model, report.step)
    restore_scaler(model, checkpoint)

    history_path = run_dir / "loss_and_lr.pth"
    history = TrainingHistory.load(history_path) if history_path.exists() else TrainingHistory()
    return report.step, history


def fine_tune(model: Any, load_path: str) -> None:
    """Start from an existing checkpoint at step 0.

    ``load_path`` may point directly at a ``.pth`` file or at a directory holding ``model.pth``;
    a plain substring test on ``".pth"`` decides which; it is not tightened to a suffix check, so
    a directory name that happens to contain ``.pth`` anywhere would be treated as a file path.
    """
    path = Path(load_path) if ".pth" in load_path else Path(load_path) / "model.pth"
    report = model.load(path, require_step=False)
    origin = "unknown step" if report.step < 0 else f"step {report.step}"
    print("!!!!! Loaded model from %s at %s (%d points)" % (path, origin, report.n_points))

    init_optimizers(model, 0)

    model.pruned_points = True


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
    dataset, eval_dataset = build_datasets(values, device)

    load_path = cli.load_path if cli.load_path is not None else values.load_path

    if cli.resume > 0:
        if cli.load_path is not None:
            print("--resume and --load_path were both given; --resume wins.")
        start_step, history = resume(model, run_dir)
    elif load_path:
        fine_tune(model, str(load_path))
        start_step, history = 0, TrainingHistory()
    else:
        init_optimizers(model, 0)
        start_step, history = 0, TrainingHistory()

    train_and_eval(start_step, model, device, dataset, eval_dataset, history, values, run_dir)
    if torch.cuda.is_available():
        print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any

import torch
from torch.optim import lr_scheduler

from models.densify import accumulate_point_grad_stats, refresh_density

__all__ = [
    "DENSIFICATION_FLAG",
    "SCHEDULE_TYPES",
    "create_learning_rate_fn",
    "init_optimizers",
    "step",
]

#: Learning-rate schedules this release implements. ``linear``, ``exp`` and ``stop`` are not
#: implemented; none is selected by any shipped config's ``training.lr.*.type``.
SCHEDULE_TYPES: tuple[str, ...] = ("none", "cosine", "cosine-hlfperiod")

#: Attribute :func:`init_optimizers` sets on the model to switch densification off, read by
#: :mod:`models.densify`. See the ``fix_keys`` note in :func:`init_optimizers`.
DENSIFICATION_FLAG = "densification_enabled"


def create_learning_rate_fn(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    args: Any,
    use_warmup: bool = True,
) -> lr_scheduler.LRScheduler | None:
    """Build the warm-up-then-decay schedule for one optimizer.

    Warm-up and decay are two separate schedulers chained by ``SequentialLR`` rather than one
    closed-form lambda, because ``CosineAnnealingLR`` is defined recursively from the *current*
    learning rate: composing it with a warm-up factor analytically would not give the same curve
    that the released checkpoints were trained on.

    Args:
        optimizer: The optimizer to schedule.
        max_steps: Total training steps; the decay is sized to fill what remains after warm-up.
        args: One ``training.lr.<group>`` node, read for ``type`` and ``warmup``.
        use_warmup: ``training.lr.use_warmup``. False collapses warm-up to a zero-length no-op
            rather than removing it, so the ``SequentialLR`` structure is the same either way.

    Returns:
        The scheduler, or ``None`` for ``type: none`` -- a fixed learning rate, which is how
        ``bkg_feats`` (base_lr 0.0) and ``mapping_mlp`` are held constant.

    Raises:
        ValueError: For any ``type`` outside :data:`SCHEDULE_TYPES`.
    """
    if args.type == "none":
        return None

    if use_warmup and int(args.warmup) > 0:
        warmup_start_factor = 1e-8
        warmup_total_iters = int(args.warmup)
    else:
        warmup_start_factor = 1.0
        warmup_total_iters = 0

    warmup_fn = lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_total_iters,
    )

    if args.type == "cosine":
        cosine_steps = max(max_steps - warmup_total_iters, 1)
    elif args.type == "cosine-hlfperiod":
        cosine_steps = max(max_steps - warmup_total_iters, 1) * 2
    else:
        raise ValueError(
            f"unsupported learning-rate schedule {args.type!r}; this release implements "
            f"{SCHEDULE_TYPES}"
        )

    decay_fn = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_fn, decay_fn], milestones=[warmup_total_iters]
    )


def _param_table(model: Any, lr_opt: Any) -> dict[str, tuple[Any, Any]]:
    """``{group name: (parameter or module, its lr config)}``.

    Every entry is fetched with ``getattr(..., None)`` and a ``None`` is skipped, which is how a
    config that disables a component (no background points, no alpha, no scaler MLP) ends up with
    no optimizer for it rather than an empty one.

    Not included, all belonging to post-paper arms this release does not build: ``refiner``,
    ``dual_head``, ``points_view_feat``, ``bkg_view_feat``, ``bkg_exp_scaler``,
    ``cubemap_textures``, ``cubemap_feature_textures``, ``cubemap_depth_textures``, ``factorized``,
    ``factorized_value`` and ``factorized_opacity``.

    PRESERVED OMISSION: ``fused_feature_encoder`` has no entry, so its parameters are never
    optimised. Harmless for every reachable config -- the four permitted
    encoding ``otype``s (``FullyFusedMLP``, ``CutlassMLP``, ``Frequency``, ``Identity``) are all
    parameter-free, and ``encoder.params`` is a genuinely zero-element tensor. A learnable encoding
    would silently stay at its initialisation; this release cannot build one.
    """
    influ_lr = lr_opt.points_influ_scores
    return {
        "points": (getattr(model, "points", None), lr_opt.points),
        "points_influ_scores": (getattr(model, "points_influ_scores", None), influ_lr),
        "points_scaler": (
            getattr(model, "points_scaler", None),
            getattr(lr_opt, "points_scaler", influ_lr),
        ),
        "points_normals": (
            getattr(model, "points_normals", None),
            getattr(lr_opt, "points_normals", lr_opt.points),
        ),
        "points_cov": (
            getattr(model, "points_cov", None),
            getattr(lr_opt, "points_cov", lr_opt.points),
        ),
        "pc_feats": (getattr(model, "pc_feats", None), lr_opt.feats),
        "points_alpha": (getattr(model, "points_alpha", None), lr_opt.points_alpha),
        "bkg_feats": (getattr(model, "bkg_feats", None), lr_opt.bkg_feats),
        "bkg_token": (getattr(model, "bkg_token", None), lr_opt.bkg_token),
        "unet": (getattr(model, "unet", None), lr_opt.unet),
        "mapping_mlp": (getattr(model, "mapping_mlp", None), lr_opt.mapping_mlp),
        "topk_mlp": (getattr(model, "topk_mlp", None), lr_opt.topk_mlp),
        "bkg_scaler": (getattr(model, "bkg_scaler", None), lr_opt.bkg_scaler),
        "bkg_score": (getattr(model, "bkg_score", None), lr_opt.bkg_score),
        "bkg_points_pc_feats": (
            getattr(model, "bkg_points_pc_feats", None),
            lr_opt.bkg_points_pc_feats,
        ),
        "bkg_points_pc_feats_mlp": (
            getattr(model, "bkg_points_pc_feats_mlp", None),
            lr_opt.bkg_points_pc_feats,
        ),
        "bkg_points_influ_scores": (
            getattr(model, "bkg_points_influ_scores", None),
            lr_opt.bkg_points_influ_scores,
        ),
        "bkg_points": (getattr(model, "bkg_points", None), lr_opt.bkg_points),
        "bkg_points_embedv": (
            getattr(model, "bkg_points_embedv", None),
            lr_opt.bkg_points_embedv,
        ),
        "append_bkg_points_embedv": (
            getattr(model, "append_bkg_points_embedv", None),
            lr_opt.bkg_points_embedv,
        ),
        "append_bkg_points_feats": (
            getattr(model, "append_bkg_points_feats", None),
            lr_opt.bkg_points_pc_feats,
        ),
        "fused_feature_mlp": (
            getattr(model, "fused_feature_mlp", None),
            lr_opt.fused_feature_mlp,
        ),
    }


def _make_optimizer(optim_type: str, params: Any, param_lr_opt: Any, lr_opt: Any):
    """Construct one ``Adam`` / ``AdamW`` over ``params``.

    ``Muon`` is not implemented: no shipped config selects it.
    """
    kwargs = dict(
        lr=param_lr_opt.base_lr * lr_opt.lr_factor,
        weight_decay=param_lr_opt.weight_decay,
        eps=lr_opt.eps,
        amsgrad=lr_opt.use_amsgrad,
        betas=(lr_opt.beta1, lr_opt.beta2),
    )
    if optim_type == "Adam":
        return torch.optim.Adam(params, **kwargs)
    if optim_type == "AdamW":
        return torch.optim.AdamW(params, **kwargs)
    raise ValueError(
        f"unsupported training.lr.optim_type {optim_type!r}; this release implements 'Adam' and "
        "'AdamW'"
    )


def init_optimizers(model: Any, total_steps: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build every optimizer and scheduler, and hang them on ``model``.

    Args:
        model: The PAPR model. ``model.optimizers`` and ``model.schedulers`` are assigned.
        total_steps: Steps already completed. Nonzero on ``--resume``; see the fast-forward note
            below and the module docstring.

    Returns:
        ``(optimizers, schedulers)``, the same dicts assigned onto the model.
    """
    optimizers: dict[str, Any] = {}
    schedulers: dict[str, Any] = {}
    model.optimizers = optimizers
    model.schedulers = schedulers

    lr_opt = model.args.training.lr
    use_warmup = lr_opt.use_warmup
    print("use_warmup: ", use_warmup)

    def _schedule(optimizer, param_lr_opt):
        return create_learning_rate_fn(
            optimizer, model.args.training.steps, param_lr_opt, use_warmup=use_warmup
        )

    for name, (param, param_lr_opt) in _param_table(model, lr_opt).items():
        if param is None:
            continue
        require_grad = param.requires_grad if isinstance(param, torch.Tensor) else True
        if not require_grad:
            continue
        opt_params = [param] if isinstance(param, torch.Tensor) else param.parameters()
        optimizers[name] = _make_optimizer(lr_opt.optim_type, opt_params, param_lr_opt, lr_opt)
        schedulers[name] = _schedule(optimizers[name], param_lr_opt)

    proximity_attn = getattr(model, "proximity_attn", None)
    if proximity_attn is not None and getattr(proximity_attn, "model_v", None) is not None:
        param_lr_opt = lr_opt.attn

        attn_v_params = list(proximity_attn.model_v.parameters())
        for name, param in proximity_attn.model_v.named_parameters():
            print("Adding param to attn_v: ", name, param.shape)

        attn_other_params = []
        model_v_param_ids = {id(p) for p in attn_v_params}
        for name, param in proximity_attn.named_parameters():
            if id(param) not in model_v_param_ids:
                print("Adding param to attn_other: ", name, param.shape)
                attn_other_params.append(param)

        if attn_v_params:
            optimizers["attn_v"] = _make_optimizer(
                lr_opt.optim_type, attn_v_params, param_lr_opt, lr_opt
            )
            schedulers["attn_v"] = _schedule(optimizers["attn_v"], param_lr_opt)

        if attn_other_params:
            optimizers["attn_other"] = _make_optimizer(
                lr_opt.optim_type, attn_other_params, param_lr_opt, lr_opt
            )
            schedulers["attn_other"] = _schedule(optimizers["attn_other"], param_lr_opt)

    fix_keys = list(model.args.training.fix_keys)
    for name in fix_keys:
        if name in optimizers:
            print("Fixing {}".format(name))
            optimizers.pop(name)
            schedulers.pop(name)
        if name == "attn":
            for alias in ("attn_v", "attn_other"):
                if alias in optimizers:
                    print("Fixing {}".format(alias))
                    optimizers.pop(alias)
                    schedulers.pop(alias)

    already_disabled = getattr(model, DENSIFICATION_FLAG, True) is False
    setattr(model, DENSIFICATION_FLAG, not fix_keys)
    if fix_keys and not already_disabled:
        print(
            "[fix_keys] Frozen parameter groups {}: point pruning and point addition are DISABLED "
            "for this run. Densification resizes parameters through their optimizers, and a frozen "
            "group has none.".format(sorted(fix_keys))
        )

    print(optimizers.keys())
    print(schedulers.keys())

    if total_steps > 0:
        for scheduler in schedulers.values():
            if scheduler is not None:
                for _ in range(total_steps):
                    scheduler.step()

    return optimizers, schedulers


def _clip_grad_norm_per_optimizer(model: Any, grad_clip_norm: float, step_idx: int) -> None:
    """Unscale and clip each optimizer's gradients separately.

    A single global ``clip_grad_norm_`` over every parameter is a NaN-contamination vector under
    AMP: one optimizer's transient NaN makes the *global* norm NaN, and the clip then writes NaN
    into every optimizer's gradients -- but the healthy optimizers' ``GradScaler`` entries were
    already marked clean at ``unscale_`` time, so they apply the poison instead of skipping. Per
    optimizer, the NaN stays inside the one optimizer that is already flagged.
    """
    norms: list[float] = []
    for optimizer in model.optimizers.values():
        if optimizer is None:
            continue
        scaler = getattr(model, "scaler", None)
        if scaler is not None and scaler.is_enabled():
            try:
                scaler.unscale_(optimizer)
            except RuntimeError:
                pass
        params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
        if not params:
            continue

        if any(torch.isnan(p.grad).any() for p in params):
            for p in params:
                p.grad.detach().zero_()
            count = getattr(model, "_nan_grad_zeroed", 0) + 1
            object.__setattr__(model, "_nan_grad_zeroed", count)
            if count % 50 == 1:
                print(f"[nan-grad-guard] zeroed NaN grads (occurrence {count}) at step {step_idx}")
            norms.append(0.0)
            continue

        norms.append(float(torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)))

    if step_idx % 201 == 0 and norms:
        print(
            f"Gradient clipping applied: per-opt norms {[round(t, 4) for t in norms]} "
            f"(clip_norm={grad_clip_norm})"
        )


def step(
    model: Any,
    step: int = -1,
    grad_clip_norm: float | None = None,
    grad_clip_value: float | None = None,
) -> bool:
    """Run one optimizer step across every group, then advance every schedule.

    Called after ``loss.backward()`` and before ``scaler.update()``.

    Args:
        model: The PAPR model.
        step: Current training step. Gates the density refresh, the ``topk_mlp`` start step and the
            periodic diagnostics.
        grad_clip_norm: ``training.grad_clip_norm``; ``<= 0`` or ``None`` disables clipping.
        grad_clip_value: ``training.grad_clip_value``; ``<= 0`` or ``None`` disables it.

    Returns:
        Whether any optimizer actually stepped. False means every group's gradients were missing --
        which is what a fully-frozen configuration looks like, and what the caller checks before
        calling ``scaler.update()``.
    """
    density_interval = getattr(model, "density_update_interval", 0)
    if density_interval > 0 and step > 0 and step % density_interval == 0:
        refresh_density(model)

    if model.points.grad is not None:
        accumulate_point_grad_stats(model)

        grad_noise_std = getattr(model, "grad_noise_std", 0.0)
        if grad_noise_std > 0:
            noise = torch.randn_like(model.points.grad) * grad_noise_std
            inv_density = 1.0 / (model.points_density.unsqueeze(-1) + 1e-8)
            model.points.grad.data += noise * inv_density

    if model.args.models.attn.use_pc_feats_directly and model.args.models.attn.use_sh:
        model.pc_feats.grad[:, 1:, :] /= model.args.models.attn.sh_N_factor

    if grad_clip_norm is not None and grad_clip_norm > 0:
        _clip_grad_norm_per_optimizer(model, grad_clip_norm, step)

    if grad_clip_value is not None and grad_clip_value > 0:
        all_params = []
        for optimizer in model.optimizers.values():
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    all_params.extend(param_group["params"])
        torch.nn.utils.clip_grad_value_(all_params, grad_clip_value)
        if step % 201 == 0:
            print(
                f"Gradient clipping applied: value {grad_clip_value:.4f} "
                f"(clip_value={grad_clip_value})"
            )

    any_stepped = False
    for opt_name, optimizer in model.optimizers.items():
        if opt_name == "topk_mlp" and step < model.args.models.topk_mlp.start_step:
            continue
        if optimizer is None:
            continue
        has_grad = any(
            param.grad is not None and param.grad.data.numel() > 0
            for param_group in optimizer.param_groups
            for param in param_group["params"]
            if param.grad is not None
        )
        if has_grad:
            model.scaler.step(optimizer)
            any_stepped = True

    for scheduler in model.schedulers.values():
        if scheduler is not None:
            scheduler.step()

    model.attn_lr = _current_lr(model, ("attn_v", "attn_other", "attn"))
    model.pts_lr = _current_lr(model, ("points",))
    return any_stepped


def _current_lr(model: Any, names: tuple[str, ...]) -> float:
    """The learning rate of the first of ``names`` that exists, for the training log."""
    for name in names:
        if name not in model.optimizers:
            continue
        scheduler = model.schedulers.get(name)
        if scheduler is not None:
            return scheduler.get_last_lr()[0]
        return model.optimizers[name].param_groups[0]["lr"]
    return 0

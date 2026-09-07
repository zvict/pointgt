from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

from utils.ply import write_ply_points

__all__ = [
    "AddedPoints",
    "PRUNABLE_PARAMS",
    "accumulate_point_grad_stats",
    "add_points",
    "add_points_knn",
    "cat_tensors_to_optimizer",
    "fresh_point_grad_stats",
    "get_prune_signal",
    "prune_optimizer",
    "prune_points",
    "refresh_density",
    "reset_point_grad_stats",
    "update_point_density",
]

#: Optimizer groups whose rows are per-point and therefore have to be resized in lockstep.
#: ``points_normals`` is deliberately absent even when the model has it: it is grown with a plain
#: ``torch.cat`` rather than through :func:`cat_tensors_to_optimizer`, so its optimizer reference
#: goes stale after an add and masking it here raises ``IndexError``. It is handled by the
#: manual fallback in :func:`prune_points` instead, which reads the up-to-date parameter.
PRUNABLE_PARAMS: tuple[str, ...] = ("points", "points_influ_scores", "points_scaler", "pc_feats")

#: Per-point tensors grown and shrunk outside the optimizer, by direct assignment, and which are
#: NOT in ``PAPR.PER_POINT_PARAMETERS`` (so they are set individually).
_EXTRA_MANUAL_PARAMS: tuple[str, ...] = ("points_cov", "points_view_feat")

#: Fallback for a model that does not publish ``PER_POINT_PARAMETERS``.
_DEFAULT_PER_POINT: tuple[str, ...] = (
    "points", "pc_feats", "points_influ_scores", "points_scaler", "points_normals",
    "points_last_grad", "points_acc_grad", "points_acc_grad_norm", "points_grad_cnt",
    "points_density",
)

#: Model attribute cleared when ``training.fix_keys`` freezes a parameter group. Both
#: ``PAPR.__init__`` and :func:`models.optim.init_optimizers` set it, from the same config key.
#: Kept as a literal here rather than imported, so ``models.densify`` does not depend on
#: ``models.optim`` (``models.optim`` imports *this* module).
_DENSIFICATION_FLAG = "densification_enabled"


def _densification_enabled(model: Any, action: str) -> bool:
    """Whether densification may run, printing the reason when it may not.

    DELIBERATE DEVIATION (release change f): freezing a parameter group with ``training.fix_keys``
    removes its optimizer, and both densification paths need that optimizer. ``PAPR.__init__``
    clears the flag (and :func:`models.optim.init_optimizers` clears it again for a model that does
    not); this refuses politely instead of crashing at the first prune.
    """
    if getattr(model, _DENSIFICATION_FLAG, True):
        return True
    print(
        f"@@@@@@@@@  {action} skipped: training.fix_keys froze at least one parameter group, so "
        "densification is disabled for this run (it resizes parameters through their optimizers)."
    )
    return False


def _per_point_names(model: Any) -> tuple[str, ...]:
    """The per-point tensor names the model wants replaced together."""
    return tuple(getattr(model, "PER_POINT_PARAMETERS", _DEFAULT_PER_POINT))


def _replace_point_parameters(model: Any, replacements: dict[str, nn.Parameter]) -> None:
    """Install a whole new set of per-point parameters.

    Routed through ``model.set_point_parameters`` when the model provides it: that setter checks
    every per-point tensor arrives at once and that they agree on the row count, which is the exact
    invariant a half-finished prune breaks -- and breaks *silently*, since a cloud where
    ``points[i]`` and ``pc_feats[i]`` describe different points still renders a plausible image.
    """
    setter = getattr(model, "set_point_parameters", None)
    if setter is not None:
        setter(replacements)
        return
    for name, tensor in replacements.items():
        setattr(model, name, tensor)


def refresh_density(model: Any) -> None:
    """Recompute ``points_density``, preferring the model's own implementation."""
    own = getattr(model, "update_point_density", None)
    if own is not None:
        own()
        return
    update_point_density(model)


class AddedPoints(NamedTuple):
    """What :func:`add_points_knn` produced."""

    #: ``(A, 3)`` new positions.
    points: Tensor
    #: ``A``, the number actually produced (may be below the requested ``add_num``).
    count: int
    #: ``(A, 1)`` influence scores for the new points.
    influ_scores: Tensor
    #: ``(A, D)`` features, or ``None`` when none were supplied.
    features: Tensor | None
    #: ``(A, 1)`` alphas, or ``None``.
    alphas: Tensor | None
    #: ``(A, 1)`` scalers, or ``None``.
    scalers: Tensor | None
    #: Indices of the *parent* points each new point was cloned from.
    source_indices: Any
    #: ``(A, 3)`` normals, or ``None``.
    normals: Tensor | None
    #: ``(A, 7)`` covariance parameters, or ``None``.
    cov: Tensor | None


def prune_optimizer(model: Any, mask: Tensor) -> dict[str, nn.Parameter]:
    """Shrink every per-point parameter to ``mask``, carrying its Adam state along.

    Args:
        model: The PAPR model. Read for ``optimizers`` and ``args.geoms.alpha.use``.
        mask: ``(M,)`` boolean, ``True`` for points to keep.

    Returns:
        ``{name: new_parameter}`` for each group that was pruned. The caller must assign these back
        onto the model -- this function repoints the *optimizer*, not the module attributes, and
        leaving the two out of step is exactly the failure it exists to prevent.
    """
    optimizable_tensors: dict[str, nn.Parameter] = {}

    params_to_prune = list(PRUNABLE_PARAMS)
    if getattr(model, "points_alpha", None) is not None and model.args.geoms.alpha.use:
        params_to_prune.append("points_alpha")

    for name, optimizer in model.optimizers.items():
        if name not in params_to_prune:
            continue

        assert len(optimizer.param_groups) == 1, (
            f"Optimizer for '{name}' should have only one parameter group"
        )
        group = optimizer.param_groups[0]
        old_param = group["params"][0]

        stored_state = optimizer.state.get(old_param, None)

        new_param = nn.Parameter(old_param.data[mask], requires_grad=old_param.requires_grad)

        if old_param.grad is not None:
            new_param.grad = old_param.grad[mask]

        if stored_state is not None:
            if "exp_avg" in stored_state:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            if "exp_avg_sq" in stored_state:
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[old_param]
            group["params"][0] = new_param
            optimizer.state[new_param] = stored_state
        else:
            group["params"][0] = new_param

        optimizable_tensors[name] = new_param

    return optimizable_tensors


def cat_tensors_to_optimizer(
    model: Any, tensors_dict: dict[str, Tensor]
) -> dict[str, nn.Parameter]:
    """Append rows to per-point parameters, extending their Adam state with zeros.

    The new rows start with zero momentum rather than inheriting their parent's. That is the right
    choice and not an oversight: a cloned point sits at exactly its parent's position, and copying
    the parent's momentum would launch both in the same direction at the same speed, which is the
    one thing that guarantees they never separate.

    Args:
        model: The PAPR model, read for ``optimizers``.
        tensors_dict: ``{name: extension}``, each ``(A, ...)`` matching the parameter's trailing
            shape.

    Returns:
        ``{name: new_parameter}``; the caller assigns them back onto the model.
    """
    optimizable_tensors: dict[str, nn.Parameter] = {}

    for name, extension_tensor in tensors_dict.items():
        assert isinstance(extension_tensor, torch.Tensor), (
            "Tensor {} is not a torch.Tensor".format(name)
        )
        if name not in model.optimizers:
            raise ValueError("Optimizer for {} not found".format(name))
        assert len(model.optimizers[name].param_groups) == 1, (
            "Optimizer for {} should have only one parameter group".format(name)
        )

        optimizer = model.optimizers[name]
        group = optimizer.param_groups[0]
        old_param = group["params"][0]

        new_param_data = torch.cat((old_param.data, extension_tensor), dim=0)
        new_param = nn.Parameter(new_param_data, requires_grad=old_param.requires_grad)

        if old_param.grad is not None:
            extension_grad = torch.zeros_like(extension_tensor)
            new_param.grad = torch.cat((old_param.grad, extension_grad), dim=0)

        stored_state = optimizer.state.get(old_param, None)
        if stored_state is not None:
            if "exp_avg" in stored_state:
                extension_state = torch.zeros_like(extension_tensor)
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], extension_state), 0)
            if "exp_avg_sq" in stored_state:
                extension_state = torch.zeros_like(extension_tensor)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], extension_state), 0
                )

            del optimizer.state[old_param]
            group["params"][0] = new_param
            optimizer.state[new_param] = stored_state
        else:
            group["params"][0] = new_param

        optimizable_tensors[name] = new_param

    return optimizable_tensors


def accumulate_point_grad_stats(model: Any) -> None:
    """Fold this step's position gradient into the four densification accumulators.

    Called from :func:`models.optim.step` before the optimizer runs, i.e. while ``points.grad``
    still holds this step's gradient.

    The ``isfinite`` guard is not cosmetic. Under AMP the ``GradScaler`` *skips* an optimizer step
    whose gradients overflowed, but these accumulators ran unconditionally in an earlier version, so
    one overflowing step wrote NaN into ``points_acc_grad_norm`` and it stayed there: every
    subsequent add sampled by a NaN key, selected a degenerate set, and eventually died in
    ``torch.cat``. Skipping the whole update on a non-finite gradient is what fixes it, and it is
    the behaviour the released checkpoints were trained with.
    """
    if model.points.grad is None:
        return
    if torch.isfinite(model.points.grad).all():
        model.points_last_grad.data = model.points.grad
        model.points_acc_grad.data += model.points.grad
        model.points_acc_grad_norm.data += torch.norm(model.points.grad, dim=-1)
        model.points_grad_cnt.data += (model.points.grad.abs().sum(-1) != 0).float()


def fresh_point_grad_stats(n: int, device: torch.device | str) -> dict[str, nn.Parameter]:
    """Zeroed accumulators (and a unit density) for an ``n``-point cloud.

    Returned rather than assigned so :func:`add_points` can hand them to the model in the same
    single, row-count-checked call as the grown parameters.
    """
    return {
        "points_last_grad": nn.Parameter(torch.zeros(n, 3, device=device), requires_grad=False),
        "points_acc_grad": nn.Parameter(torch.zeros(n, 3, device=device), requires_grad=False),
        "points_acc_grad_norm": nn.Parameter(torch.zeros(n, device=device), requires_grad=False),
        "points_grad_cnt": nn.Parameter(torch.zeros(n, device=device), requires_grad=False),
        "points_density": nn.Parameter(torch.ones(n, device=device), requires_grad=False),
    }


def reset_point_grad_stats(model: Any) -> None:
    """Zero all four accumulators and resize them to the current point count.

    Called after an add. The statistics are reset rather than extended because the new points have
    no history and the old points' history was gathered under a different cloud -- the sampling
    keys are only comparable within one densification interval.
    """
    fresh = fresh_point_grad_stats(model.points.shape[0], model.points.device)
    if getattr(model, "points_density", None) is None:
        fresh.pop("points_density")
    for name, tensor in fresh.items():
        setattr(model, name, tensor)


@torch.no_grad()
def update_point_density(model: Any) -> None:
    """Recompute each point's local density as ``1 / distance to its k-th neighbour``.

    Only consumed by the ``grad_noise_std`` gradient perturbation, which is 0.0 in every shipped
    config -- so on the released configs this is bookkeeping with no effect on the trained weights.
    It is kept because ``points_density`` is a per-point tensor that appears in archived
    ``state_dict``s and must stay row-aligned with ``points`` for those checkpoints to load.
    """
    if getattr(model, "points_density", None) is None:
        return

    from scipy.spatial import cKDTree

    n_points = model.points.shape[0]
    if n_points < model.density_k:
        model.points_density.data.fill_(1.0)
        return

    device = model.points.device
    points_np = (model.points.detach() / model.coord_scale).cpu().numpy()

    tree = cKDTree(points_np)
    distances, _ = tree.query(points_np, k=model.density_k, workers=-1)

    kth_distances = np.maximum(distances[:, -1], 1e-10)
    density_np = 1.0 / kth_distances

    model.points_density.data = torch.from_numpy(density_np).float().to(device)

    density = model.points_density
    print(
        f"[Density] Updated point density: min={density.min().item():.4f}, "
        f"max={density.max().item():.4f}, mean={density.mean().item():.4f}, "
        f"std={density.std().item():.4f}"
    )


def get_prune_signal(model: Any) -> Tensor | None:
    """The per-point scalar the prune threshold is compared against.

    Args:
        model: The PAPR model, read for ``args.training.prune_by`` and ``points_influ_scores``.

    Returns:
        ``(M, 1)`` influence scores.

    Raises:
        ValueError: For any other ``prune_by``. Pruning by the learned alpha or by factorized-table
            statistics is not supported; neither the alpha stack nor the factorized renderer is
            part of this release.
    """
    prune_by = model.args.training.prune_by
    if prune_by in ("influ_score", "influence"):
        return model.points_influ_scores
    raise ValueError(
        f"unsupported training.prune_by {prune_by!r}; this release prunes by 'influ_score'"
    )


def prune_points(model: Any, thresh: float, step: int = -1) -> int:
    """Delete points whose prune signal falls the wrong side of ``thresh``.

    Args:
        model: The PAPR model. Mutates its per-point parameters and its optimizers.
        thresh: The threshold; the caller computes it (fixed, interpolated or from a quantile).
        step: Current training step, consulted only by the ``prune_max_num`` rate limit.

    Returns:
        How many points were removed.
    """
    num_pruned = 0
    if not _densification_enabled(model, "prune_points"):
        return num_pruned

    prune_by = get_prune_signal(model)
    print(f"prune_by: {model.args.training.prune_by}")

    if prune_by is None:
        return num_pruned

    print(
        "@@@@@@@@@  before prune: ", thresh, model.points.shape, prune_by.shape,
        prune_by.min().item(), prune_by.max().item(),
    )
    training = model.args.training
    if training.prune_type == "<":
        mask = prune_by[:, 0] > thresh
    elif training.prune_type == ">":
        mask = prune_by[:, 0] < thresh
    else:
        raise ValueError(
            f"unknown training.prune_type {training.prune_type!r}; expected '<' or '>'"
        )

    num_to_prune = torch.sum(mask == 0)
    if num_to_prune > training.prune_max_num and step < training.prune_max_num_step:
        if training.prune_type == "<":
            _, indices = torch.topk(
                prune_by[:, 0], training.prune_max_num, largest=False, sorted=False
            )
        else:
            _, indices = torch.topk(
                prune_by[:, 0], training.prune_max_num, largest=True, sorted=False
            )
        mask = torch.ones_like(prune_by[:, 0], dtype=torch.bool)
        mask[indices] = False

    num_pruned = torch.sum(mask == 0).item()
    if num_pruned > 0:
        optimizable_tensors = prune_optimizer(model, mask)
        replacements: dict[str, nn.Parameter] = {}

        def keep(name: str) -> nn.Parameter:
            """The pruned parameter: from the optimizer when it has one, by hand when it does not.

            A per-point parameter has no optimizer when `fix_keys` froze it or when the config
            turned its learning off (`points_scaler` with `scaler_learn: false`). It still has to
            shrink, or the next gather pairs its rows with the wrong points.
            """
            if name in optimizable_tensors:
                return optimizable_tensors[name]
            current = getattr(model, name)
            return nn.Parameter(current[mask], requires_grad=current.requires_grad)

        for name in ("points", "pc_feats", "points_influ_scores", "points_scaler"):
            replacements[name] = keep(name)
        if getattr(model, "points_alpha", None) is not None:
            model.points_alpha = optimizable_tensors["points_alpha"]

        per_point = _per_point_names(model)

        for name in tuple(n for n in per_point if n not in replacements) + _EXTRA_MANUAL_PARAMS:
            param = getattr(model, name, None)
            if param is None:
                continue
            if param.shape[0] != mask.sum():
                pruned = nn.Parameter(param[mask], requires_grad=param.requires_grad)
            else:
                pruned = param
            if name in per_point:
                replacements[name] = pruned
            else:
                setattr(model, name, pruned)

        _replace_point_parameters(model, replacements)

    print("@@@@@@@@@  pruned {}/{}".format(num_pruned, mask.shape[0]))

    if num_pruned > 0:
        refresh_density(model)
    return num_pruned


def add_points_knn(
    coords: Tensor,
    influ_scores: Tensor,
    add_num: int,
    *,
    comb_type: str = "clone",
    sample_type: str = "acc-coord-grad-norm-cnt-max",
    point_features: Tensor | None = None,
    point_alphas: Tensor | None = None,
    point_scalers: Tensor | None = None,
    acc_coord_grad_norm: np.ndarray | None = None,
    grad_cnt: np.ndarray | None = None,
    point_normals: Any = None,
    point_cov: Any = None,
) -> AddedPoints:
    """Choose where to densify and produce the new points.

    Two decisions, and only one combination of them is reachable across the shipped configs:

    * **where** (``sample_type``): ``acc-coord-grad-norm-cnt-max`` ranks points by accumulated
      position-gradient magnitude divided by ``grad_cnt + 1``. The division is what makes it a rate
      rather than a total -- without it the ranking just finds the points that were visible most
      often. The ``+1`` both avoids a divide-by-zero for never-seen points and damps the rate of
      points seen only once or twice.
    * **how** (``comb_type``): ``clone`` copies the parent's position, features, influence and
      scaler exactly. The two land on top of each other and are separated only by their subsequent
      gradients -- which differ, because ``cat_tensors_to_optimizer`` gives the child zero momentum.

    Other placements (KD-tree neighbour means, random barycentric blends, inverse-distance
    weightings, moves along the gradient) and other rankings are not implemented: they are
    unreachable, and the KD-tree ones would put a ``scipy`` query on the densification path.

    Args:
        coords: ``(M, 3)`` current positions, on CPU.
        influ_scores: ``(M, 1)`` influence scores, on CPU.
        add_num: How many points to add. When the cloud is smaller than this, every point is cloned
            and fewer than ``add_num`` are produced.
        comb_type: Must be ``"clone"``.
        sample_type: Must be ``"acc-coord-grad-norm-cnt-max"``.
        point_features: ``(M, D)`` features to clone, or ``None``.
        point_alphas: ``(M, 1)`` alphas to clone, or ``None``.
        point_scalers: ``(M, 1)`` scalers to clone, or ``None``.
        acc_coord_grad_norm: ``(M,)`` accumulated gradient norms, as numpy.
        grad_cnt: ``(M,)`` per-point gradient counts, as numpy.
        point_normals: ``(M, 3)`` normals to clone, or ``None``.
        point_cov: ``(M, 7)`` covariance parameters to clone, or ``None``.

    Returns:
        An :class:`AddedPoints`.
    """
    if comb_type != "clone":
        raise ValueError(
            f"unsupported geoms.points.add_type {comb_type!r}; this release implements 'clone'"
        )
    if sample_type != "acc-coord-grad-norm-cnt-max":
        raise ValueError(
            f"unsupported geoms.points.add_sample_type {sample_type!r}; this release implements "
            "'acc-coord-grad-norm-cnt-max'"
        )

    n = coords.shape[0]

    if n <= add_num:
        query_coords = coords
        inds: Any = list(range(n))
    else:
        inds = np.argsort(acc_coord_grad_norm / (grad_cnt + 1))[-add_num:]
        query_coords = coords[inds, :]

    new_coords = query_coords
    new_influ_scores = influ_scores[inds, :]
    new_features = None if point_features is None else point_features[inds, :]
    new_alphas = None if point_alphas is None else point_alphas[inds, :]
    new_scalers = None if point_scalers is None else point_scalers[inds, :]

    new_point_normals = None
    if point_normals is not None:
        arr = point_normals if isinstance(point_normals, np.ndarray) else point_normals.numpy()
        new_point_normals = torch.from_numpy(arr[inds, :]).float()

    new_point_cov = None
    if point_cov is not None:
        arr = point_cov if isinstance(point_cov, np.ndarray) else point_cov.numpy()
        new_point_cov = torch.from_numpy(arr[inds, :]).float()

    return AddedPoints(
        new_coords,
        len(new_coords),
        new_influ_scores,
        new_features,
        new_alphas,
        new_scalers,
        inds,
        new_point_normals,
        new_point_cov,
    )


def _print_quantiles(label: str, values: np.ndarray, value_fmt: str = "{:.6f}") -> None:
    """Log the quartiles of a densification statistic."""
    quantiles = [0, 0.25, 0.5, 0.75, 1.0]
    computed = np.quantile(values, quantiles)
    print("@@@@@@@ {} quantiles:".format(label))
    for i, q in enumerate(quantiles):
        print(("  quantile {:.2f}: " + value_fmt).format(q, computed[i]))


def add_points(model: Any, add_num: int, step: int = -1, prune_thresh: float = 0.0) -> int:
    """Densify the cloud by ``add_num`` points and keep every per-point tensor in step.

    Args:
        model: The PAPR model. Mutates its per-point parameters and its optimizers.
        add_num: How many to add. The caller has already clamped this against ``max_num_pts``.
        step: Current training step, used only to name the debug point-cloud dump.
        prune_thresh: Accepted and unused; the caller passes it so the two call sites stay
            symmetric with :func:`prune_points`.

    Returns:
        How many points were actually added.
    """
    if not _densification_enabled(model, "add_points"):
        return 0

    points = model.points.detach().cpu()

    print(
        "@@@@@@@@@  last_coord_grad: ", model.points_last_grad.shape,
        model.points_last_grad.min().item(), model.points_last_grad.max().item(),
    )
    print(
        "@@@@@@@@@  acc_coord_grad: ", model.points_acc_grad.shape,
        model.points_acc_grad.min().item(), model.points_acc_grad.max().item(),
    )
    print(
        "@@@@@@@@@  acc_coord_grad_norm: ", model.points_acc_grad_norm.shape,
        model.points_acc_grad_norm.min().item(), model.points_acc_grad_norm.max().item(),
    )
    print(
        "@@@@@@@@@  grad_cnt: ", model.points_grad_cnt.shape,
        model.points_grad_cnt.min().item(), model.points_grad_cnt.max().item(),
    )

    acc_coord_grad = model.points_acc_grad.squeeze().detach().cpu().numpy()
    acc_coord_grad_norm = model.points_acc_grad_norm.squeeze().detach().cpu().numpy()
    grad_cnt = model.points_grad_cnt.squeeze().detach().cpu().numpy()

    norm_acc_coord_grad = np.linalg.norm(acc_coord_grad, axis=1)
    _print_quantiles("norm acc_coord_grad", norm_acc_coord_grad)
    _print_quantiles("acc_coord_grad_norm", acc_coord_grad_norm)
    _print_quantiles("grad_cnt", grad_cnt)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_grad = np.divide(
            norm_acc_coord_grad, grad_cnt.squeeze(),
            out=np.zeros_like(norm_acc_coord_grad), where=grad_cnt.squeeze() != 0,
        )
        ratio_grad_norm = np.divide(
            acc_coord_grad_norm, grad_cnt.squeeze(),
            out=np.zeros_like(acc_coord_grad_norm), where=grad_cnt.squeeze() != 0,
        )
    _print_quantiles("acc_coord_grad / grad_cnt", ratio_grad, value_fmt="{}")
    _print_quantiles("acc_coord_grad_norm / grad_cnt", ratio_grad_norm)

    point_features = model.pc_feats.detach().cpu()
    points_alpha = getattr(model, "points_alpha", None)
    point_alphas = None if points_alpha is None else points_alpha.detach().cpu()
    point_scalers = model.points_scaler.detach().cpu()
    points_normals = getattr(model, "points_normals", None)
    points_cov = getattr(model, "points_cov", None)

    added = add_points_knn(
        points,
        model.points_influ_scores.detach().cpu(),
        add_num,
        comb_type=model.args.geoms.points.add_type,
        sample_type=model.args.geoms.points.add_sample_type,
        point_features=point_features,
        point_alphas=point_alphas,
        point_scalers=point_scalers,
        acc_coord_grad_norm=acc_coord_grad_norm,
        grad_cnt=grad_cnt,
        point_normals=None if points_normals is None else points_normals.detach().cpu(),
        point_cov=None if points_cov is None else points_cov.detach().cpu(),
    )
    num_new_points = added.count
    if num_new_points <= 0:
        return num_new_points

    new_scalers = added.scalers
    if isinstance(new_scalers, np.ndarray):
        new_scalers = torch.from_numpy(new_scalers).float()
    elif new_scalers is None:
        scaler_init_val = getattr(model.args.geoms.points, "scaler_init_val", 1.0)
        new_scalers = torch.ones(num_new_points, 1) * scaler_init_val

    if model.args.geoms.points.save_added_points:
        from utils.session import make_run_dir

        save_dir = make_run_dir(model.args) / "point_clouds"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "added_points_{}.ply".format(step)
        pts_np = points.detach().cpu().numpy()
        colors = np.ones_like(pts_np) * 0.5
        colors[added.source_indices, :] = np.array([1, 0, 0])
        write_ply_points(save_path, pts_np, colors=colors)
        print("@@@@@@@@@  saved added points to {}".format(save_path))

    extensions = {
        "points": added.points.to(model.points.device),
        "pc_feats": added.features.to(model.pc_feats.device),
    }
    if "points_influ_scores" in model.optimizers:
        extensions["points_influ_scores"] = added.influ_scores.to(
            model.points_influ_scores.device
        )
    points_scaler_new = new_scalers.to(model.points_scaler.device)
    if "points_scaler" in model.optimizers:
        extensions["points_scaler"] = points_scaler_new
    if getattr(model, "points_alpha", None) is not None:
        extensions["points_alpha"] = added.alphas.to(model.points_alpha.device)

    optimizable_tensors = cat_tensors_to_optimizer(model, extensions)

    manual_extension = {
        "points_influ_scores": added.influ_scores.to(model.points_influ_scores.device),
        "points_scaler": points_scaler_new,
    }

    def grown(name: str) -> nn.Parameter:
        """The grown parameter: from the optimizer when it has one, by hand when it does not."""
        if name in optimizable_tensors:
            return optimizable_tensors[name]
        current = getattr(model, name)
        return nn.Parameter(
            torch.cat([current, manual_extension[name]], dim=0),
            requires_grad=current.requires_grad,
        )

    replacements: dict[str, nn.Parameter] = {
        "points": optimizable_tensors["points"],
        "pc_feats": optimizable_tensors["pc_feats"],
        "points_influ_scores": grown("points_influ_scores"),
        "points_scaler": grown("points_scaler"),
    }
    if getattr(model, "points_alpha", None) is not None:
        model.points_alpha = optimizable_tensors["points_alpha"]

    per_point = _per_point_names(model)

    if points_normals is not None:
        new_point_normals = added.normals
        if new_point_normals is None:
            new_point_normals = torch.randn(num_new_points, 3, device=points_normals.device)
        elif isinstance(new_point_normals, np.ndarray):
            new_point_normals = torch.from_numpy(new_point_normals).float()
        if new_point_normals.ndim != 2 or new_point_normals.shape[0] != num_new_points:
            if new_point_normals.numel() == num_new_points * 3:
                new_point_normals = new_point_normals.reshape(num_new_points, 3)
            else:
                new_point_normals = torch.randn(num_new_points, 3, device=points_normals.device)
        grown_normals = nn.Parameter(
            torch.cat([points_normals, new_point_normals.to(points_normals.device)], dim=0),
            requires_grad=points_normals.requires_grad,
        )
        if "points_normals" in per_point:
            replacements["points_normals"] = grown_normals
        else:
            model.points_normals = grown_normals

    if points_cov is not None and added.cov is not None:
        new_point_cov = added.cov
        if isinstance(new_point_cov, np.ndarray):
            new_point_cov = torch.from_numpy(new_point_cov).float()
        model.points_cov = nn.Parameter(
            torch.cat([points_cov, new_point_cov.to(points_cov.device)], dim=0),
            requires_grad=points_cov.requires_grad,
        )

    total_points = replacements["points"].shape[0]
    fresh = fresh_point_grad_stats(total_points, model.points.device)
    for name, tensor in fresh.items():
        if name in per_point:
            replacements[name] = tensor
        elif getattr(model, name, None) is not None:
            setattr(model, name, tensor)

    _replace_point_parameters(model, replacements)

    print("@@@@@@@@@  added {} points".format(num_new_points))

    refresh_density(model)
    return num_new_points

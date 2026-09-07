from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from pytorch3d.loss import chamfer_distance
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module import-light
    from uv.nuvo import ChartOutputs, Nuvo

__all__ = [
    "cluster_loss",
    "entropy_loss",
    "geometry_losses",
    "jacobian_distortion_losses",
    "surface_loss",
    "three_two_three_loss",
    "two_three_two_loss",
    "uv_range_loss",
]


def three_two_three_loss(points_3d: Tensor, chart_probs: Tensor,
                         points_3d_from_pred_uv: list[Tensor]) -> Tensor:
    """Squared reconstruction error of ``x -> uv -> x``, weighted by chart probability.

    Args:
        points_3d: ``(B, 3)`` surface points.
        chart_probs: ``(B, num_charts)``.
        points_3d_from_pred_uv: Per chart, ``(B, 3)`` -- the point each chart reconstructs.

    Returns:
        A ``(1,)`` tensor: the sum over charts of the probability-weighted mean squared distance.
    """
    loss = torch.zeros(1, device=points_3d.device)
    for chart_idx, reconstructed in enumerate(points_3d_from_pred_uv):
        loss = loss + (chart_probs[:, chart_idx]
                       * ((points_3d - reconstructed).norm(dim=1).pow(2))).mean()
    return loss


def two_three_two_loss(uvs: Tensor, points_2d_from_sampled_uv: list[Tensor]) -> Tensor:
    """Squared reconstruction error of ``uv -> x -> uv``, over uniform samples of the unit square.

    Unweighted by chart probability, unlike :func:`three_two_three_loss`: the round trip must hold
    everywhere in the square, including where this chart owns nothing, or the chart is free to leave
    most of its texture unused.

    Args:
        uvs: ``(B, 2)`` uniform samples in ``[0, 1]^2``.
        points_2d_from_sampled_uv: Per chart, ``(B, 2)``.

    Returns:
        A ``(1,)`` tensor.
    """
    loss = torch.zeros(1, device=uvs.device)
    for per_chart in points_2d_from_sampled_uv:
        loss = loss + (uvs - per_chart).norm(dim=1).pow(2).mean()
    return loss


def entropy_loss(chart_probs_points_3d_from_sampled_uv: list[Tensor]) -> Tensor:
    """Negative log-likelihood that chart ``i`` claims the points chart ``i`` generated.

    The ``+1e-6`` inside the log is what keeps this finite when a chart is confidently wrong -- the
    exact case the loss exists to fix -- so it caps the per-point penalty at ``13.8`` instead of
    letting one point produce an infinite gradient.

    Args:
        chart_probs_points_3d_from_sampled_uv: Per chart, ``(B, num_charts)``: the assignment
            network's opinion of the points that chart generated from random UVs.

    Returns:
        A ``(1,)`` tensor.
    """
    device = chart_probs_points_3d_from_sampled_uv[0].device
    loss = torch.zeros(1, device=device)
    for chart_idx, probs in enumerate(chart_probs_points_3d_from_sampled_uv):
        loss = loss + -torch.log(probs[:, chart_idx] + 1e-6).mean()
    return loss


def surface_loss(points_3d: Tensor, points_3d_from_sampled_uv: list[Tensor]) -> Tensor:
    """Symmetric chamfer between the real surface points and everything the charts generate.

    Both directions matter and pytorch3d's default gives both: one keeps the charts on the surface,
    the other keeps them covering all of it. This is the anchor of the whole objective -- at weight
    10 against 1 for the cycles -- because every other term is satisfiable by a self-consistent
    atlas of the wrong shape.

    All charts' generated points are concatenated into one cloud, so the term says nothing about
    *which* chart covers what; that is the cluster and entropy losses' job.

    Args:
        points_3d: ``(B, 3)``.
        points_3d_from_sampled_uv: Per chart, ``(B, 3)``.

    Returns:
        A scalar tensor.
    """
    loss, _ = chamfer_distance(points_3d.unsqueeze(0),
                               torch.cat(points_3d_from_sampled_uv, dim=0).unsqueeze(0))
    return loss


def cluster_loss(points_3d: Tensor, chart_probs: Tensor) -> Tensor:
    """Probability-weighted variance of each chart around its own soft centroid.

    The centroid is the probability-weighted mean of the points, so this is a soft K-means energy
    over the assignment: charts that spread across the object pay, charts that own one connected
    region do not.

    Note that the sum is divided by the point count but *not* by the number of charts, and the
    centroid denominator is not clamped -- a chart that no point claims divides by ~0 and produces a
    non-finite centroid. It cannot happen from a softmax over a real batch, and it is preserved.

    Args:
        points_3d: ``(B, 3)``.
        chart_probs: ``(B, num_charts)``.

    Returns:
        A scalar tensor.
    """
    numerators = torch.matmul(chart_probs.t(), points_3d)
    denominators = chart_probs.sum(dim=0)
    centroids = numerators / denominators[:, None]
    squared_dists = torch.cdist(points_3d, centroids).pow(2)
    return (squared_dists * chart_probs / points_3d.shape[0]).sum()


def jacobian_distortion_losses(texture_coordinate_mlp: Any, num_charts: int,
                               points_3d: Tensor) -> dict[str, Tensor]:
    """Distortion of the ``x -> uv`` map, from the singular values of its Jacobian.

    This is the paper's **normal-free** distortion term, and that
    is its whole reason for existing: the original conformal loss measures the angle between two
    tangent vectors pushed through the map, which needs a surface normal at every point. Point
    clouds do not come with reliable normals, and estimating them adds a preprocessing step whose
    errors show up directly in the atlas. Differentiating the map itself needs nothing but the map.

    Two ``vJP``s give the rows of the ``2x3`` Jacobian without ever materialising it, then
    ``J J^T`` is a ``2x2`` whose eigenvalues are closed-form. Three distortion measures come out:

    * ``isotropy`` -- ``(s1 - s2)^2``, zero when the map scales both surface directions equally.
      This is the *conformal* (angle-preserving) term, and the only one any shipped config weights.
    * ``area_dirichlet`` -- ``(s1 s2 - 1)^2``, zero at unit area scaling.
    * ``area_logdet`` -- ``log(s1 s2)^2``, the same idea in log space, so it penalises a chart that
      shrinks by 10x as much as one that grows by 10x.

    All three are always computed; the caller weights them. This costs nothing, since the Jacobian
    is the expensive part.

    The ``clamp(..., min=1e-8)`` before each square root is load-bearing: ``trace^2 - 4 det`` is
    algebraically non-negative but is a difference of similar magnitudes in floating point, and a
    tiny negative there makes the gradient NaN rather than the value.

    Args:
        texture_coordinate_mlp: The ``x -> uv`` map, called per chart via ``mlp_idx``.
        num_charts: How many charts to average over.
        points_3d: ``(B, 3)``. **Must be a leaf with ``requires_grad=True``** -- this differentiates
            with respect to it, and a detached input raises rather than returning zero.

    Returns:
        ``{"isotropy", "area_dirichlet", "area_logdet"}``, each a scalar averaged over charts.
    """
    if not points_3d.requires_grad:
        raise ValueError(
            "jacobian_distortion_losses differentiates the UV map with respect to points_3d; pass "
            "a tensor with requires_grad=True (the training step sets it on the sampled batch)"
        )

    chart_losses_isotropy: list[Tensor] = []
    chart_losses_area_dirichlet: list[Tensor] = []
    chart_losses_area_logdet: list[Tensor] = []

    for chart_idx in range(num_charts):
        uv_coords = texture_coordinate_mlp(points_3d, mlp_idx=chart_idx)
        u_coords = uv_coords[:, 0]
        v_coords = uv_coords[:, 1]

        grad_outputs = torch.ones_like(u_coords)
        grad_u = torch.autograd.grad(u_coords, points_3d, grad_outputs=grad_outputs,
                                     create_graph=True, retain_graph=True)[0]
        grad_v = torch.autograd.grad(v_coords, points_3d, grad_outputs=grad_outputs,
                                     create_graph=True)[0]

        jjt_11 = torch.sum(grad_u * grad_u, dim=1)
        jjt_12 = torch.sum(grad_u * grad_v, dim=1)
        jjt_22 = torch.sum(grad_v * grad_v, dim=1)

        trace = jjt_11 + jjt_22
        det = jjt_11 * jjt_22 - jjt_12 * jjt_12
        sqrt_term = torch.sqrt(torch.clamp(trace.pow(2) - 4 * det, min=1e-8))

        eigval_sq_1 = 0.5 * (trace + sqrt_term)
        eigval_sq_2 = 0.5 * (trace - sqrt_term)
        sigma_1 = torch.sqrt(torch.clamp(eigval_sq_1, min=1e-8))
        sigma_2 = torch.sqrt(torch.clamp(eigval_sq_2, min=1e-8))

        chart_losses_isotropy.append(torch.pow(sigma_1 - sigma_2, 2).mean())
        chart_losses_area_dirichlet.append(torch.pow(sigma_1 * sigma_2 - 1.0, 2).mean())
        chart_losses_area_logdet.append(
            torch.pow(torch.log(torch.clamp(sigma_1 * sigma_2, min=1e-8)), 2).mean())

    return {
        "isotropy": torch.stack(chart_losses_isotropy).mean(),
        "area_dirichlet": torch.stack(chart_losses_area_dirichlet).mean(),
        "area_logdet": torch.stack(chart_losses_area_logdet).mean(),
    }


def uv_range_loss(uv: Tensor, uv_min: float = 0.0, uv_max: float = 1.0) -> Tensor:
    """One-sided hinge pulling UVs back into ``[uv_min, uv_max]``.

    Zero and gradient-free inside the range, linear outside, so it does nothing to a chart that
    behaves and pulls one that has drifted out of the texture's support -- where the zero-padded
    bilinear lookup gives it no gradient of its own.

    Args:
        uv: ``(B, 2)`` predicted coordinates.
        uv_min / uv_max: The valid square.

    Returns:
        A scalar: the mean one-sided excess over both coordinates.
    """
    return (torch.relu(uv - uv_max) + torch.relu(uv_min - uv)).mean()


def geometry_losses(model: "Nuvo", step: int, points_3d: Tensor,
                    charts: "ChartOutputs") -> dict[str, Tensor]:
    """Every geometry term and their weighted sum.

    A term whose weight is ``<= 0`` is not computed at all -- it is not merely multiplied by zero --
    so switching one off makes the step cheaper as well as changing the objective. The three
    Jacobian measures are computed together whenever *any* of their weights is positive.

    Args:
        model: The atlas, for its config, its chart count and its UV map.
        step: Iteration, used only for the 200-step progress print.
        points_3d: ``(B, 3)`` surface points; must require grad if a Jacobian weight is positive.
        charts: The output of :meth:`uv.nuvo.Nuvo.get_all_charts` for these points.

    Returns:
        ``{"loss_combined": ..., "<term>": ...}`` with the unweighted value of every term.
        ``conformal`` and ``stretch`` are not ported and a config that asks for them is refused at
        construction, so those keys are absent rather than permanently zero.
    """
    conf = model.conf
    device = points_3d.device
    zero = torch.tensor(0.0, device=device)

    loss_three_two_three = (
        three_two_three_loss(points_3d, charts.chart_probs, charts.points_3d_from_pred_uv)
        if conf.loss.three_two_three > 0 else zero)
    loss_two_three_two = (
        two_three_two_loss(charts.random_uvs, charts.points_2d_from_sampled_uv)
        if conf.loss.two_three_two > 0 else zero)
    loss_entropy = (
        entropy_loss(charts.chart_probs_points_3d_from_sampled_uv)
        if conf.loss.entropy > 0 else zero)
    loss_surface = (
        surface_loss(points_3d, charts.points_3d_from_sampled_uv)
        if conf.loss.surface > 0 else zero)
    loss_cluster = (
        cluster_loss(points_3d, charts.chart_probs) if conf.loss.cluster > 0 else zero)

    weight_dirichlet = conf.loss.get("jacobian_area_dirichlet", 0)
    weight_logdet = conf.loss.get("jacobian_area_logdet", 0)
    if conf.loss.jacobian_conformal > 0 or weight_dirichlet > 0 or weight_logdet > 0:
        jac = jacobian_distortion_losses(model.texture_coordinate_mlp, model.num_charts, points_3d)
        loss_jacobian_conformal = jac["isotropy"]
        loss_jacobian_area_dirichlet = jac["area_dirichlet"]
        loss_jacobian_area_logdet = jac["area_logdet"]
    else:
        loss_jacobian_conformal = zero
        loss_jacobian_area_dirichlet = zero
        loss_jacobian_area_logdet = zero

    if step % 200 == 0:
        print(" ***** loss_three_two_three:", loss_three_two_three.item(),
              "loss_two_three_two:", loss_two_three_two.item(),
              "loss_entropy:", loss_entropy.item(), "loss_surface:", loss_surface.item())
        print(" ***** loss_cluster:", loss_cluster.item(),
              "loss_jacobian_conformal:", loss_jacobian_conformal.item())
        print(" ***** loss_jacobian_area_dirichlet:", loss_jacobian_area_dirichlet.item(),
              "loss_jacobian_area_logdet:", loss_jacobian_area_logdet.item())

    loss = (
        conf.loss.three_two_three * loss_three_two_three
        + conf.loss.two_three_two * loss_two_three_two
        + conf.loss.entropy * loss_entropy
        + conf.loss.surface * loss_surface
        + conf.loss.cluster * loss_cluster
        + conf.loss.jacobian_conformal * loss_jacobian_conformal
        + weight_dirichlet * loss_jacobian_area_dirichlet
        + weight_logdet * loss_jacobian_area_logdet
    )
    return {
        "loss_combined": loss,
        "three_two_three": loss_three_two_three,
        "two_three_two": loss_two_three_two,
        "entropy": loss_entropy,
        "surface": loss_surface,
        "cluster": loss_cluster,
        "jacobian_conformal": loss_jacobian_conformal,
        "jacobian_area_dirichlet": loss_jacobian_area_dirichlet,
        "jacobian_area_logdet": loss_jacobian_area_logdet,
    }

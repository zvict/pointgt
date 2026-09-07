from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from models.surface import get_surface_points_from_rays
from utils.ply import write_ply_points

__all__ = [
    "SurfacePointSet",
    "chart_assignment_clusters",
    "estimate_normals",
    "extract_surface_points",
    "farthest_point_sample",
    "filter_noisy_points",
    "kmeans_lloyd",
    "knn_voting_cleanup",
    "render_view_surface_points",
    "warmup_chart_assignment",
]

#: The fusion the surface sampler uses, hardcoded at the call site rather than read from config.
#: ``nuvo.sample.sp_fuse_type`` reads "projection" in the shipped configs and is NOT what is
#: actually used here. Preserved, including the trap.
SURFACE_FUSE_TYPE = "position-sphere-projection"


class SurfacePointSet(NamedTuple):
    """The cloud stage 1 optimises over, and the colours stage 2 fits."""

    #: ``(M, 3)`` surface points in scene units (already divided by ``coord_scale``). The first
    #: ``num_original_surface_points`` rows are rendered surface points; anything after them is the
    #: model's own point cloud, appended for the geometry loss only.
    surface_points: Tensor
    #: ``(num_original_surface_points, 3)`` observed colour of each rendered surface point.
    rgb: Tensor
    #: Where the rendered points end and the appended cloud begins.
    num_original_surface_points: int


@torch.no_grad()
def render_view_surface_points(model: Any, dataset: Any, view_idx: int, *, tile_height: int,
                               tile_width: int, divide_by_coord_scale: bool = True) -> Tensor:
    """Surface point for every pixel of one view. ``(1, H, W, 3)``.

    Tiled because the attention over a full frame does not fit in memory: ``H * W * K`` candidate
    points at once is tens of gigabytes at 800x800. Unlike the image renderer there is no decoder to
    keep whole, so the tiles are genuinely independent and the seams are exact.

    Args:
        model: A trained PAPR model with its camera set.
        dataset: The split to read rays, poses and pixel coordinates from.
        view_idx: Which view.
        tile_height / tile_width: Tile size, from ``eval.max_height`` / ``eval.max_width``.
        divide_by_coord_scale: ``texture.divide_by_coord_scale``; true everywhere.

    Returns:
        ``(1, H, W, 3)`` in scene units.
    """
    height, width = dataset.H, dataset.W
    points = model.get_points().points
    pc_feats = model.get_pc_feats()
    influ_scores = model.get_influ_scores()
    scaler = model.get_scaler()
    bkg_feats = model.get_append_bkg_points_feats()

    rays_o = dataset.rayo[view_idx:view_idx + 1]
    rays_d = dataset.rayd[view_idx:view_idx + 1]
    pix_coords = dataset.pix_coords.unsqueeze(0)
    c2w = dataset.c2w[view_idx:view_idx + 1]

    surface_points = torch.empty(1, height, width, 3, device=points.device)
    for top in range(0, height, tile_height):
        for left in range(0, width, tile_width):
            bottom, right = min(top + tile_height, height), min(left + tile_width, width)
            selection = model.select_topk(points, c2w, pix_coords[:, top:bottom, left:right])
            sample = get_surface_points_from_rays(
                model.proximity_attn,
                points,
                pc_feats,
                selection.indices,
                rays_o,
                rays_d[:, top:bottom, left:right],
                append_bkg_points_feats=bkg_feats,
                points_influ_scores=influ_scores,
                points_scaler=scaler,
                influ_fuse_type=model.args.influ_scores_fuse_type,
                attn_temp=model.args.attn_act_temp,
                fuse_type=SURFACE_FUSE_TYPE,
                coord_scale=model.coord_scale,
                divide_by_coord_scale=divide_by_coord_scale,
            )
            surface_points[:, top:bottom, left:right, :] = sample.surface_points
    return surface_points


def filter_noisy_points(method: str, surface_points: Tensor, rgb: Tensor, model: Any,
                        args: Any) -> tuple[Tensor, Tensor]:
    """Drop surface points that are not on the surface.

    Two independent tests, either or both:

    * ``"stats"`` (the shipped setting) -- open3d's statistical outlier removal: a point whose mean
      distance to its ``nb_neighbors`` nearest neighbours is more than ``std_ratio`` standard
      deviations above the cloud's average is dropped. Catches the isolated points that grazing rays
      and semi-transparent edges produce.
    * ``"knn"`` -- drop any point further than ``0.003`` scene units from the model's own cloud. A
      much blunter test that assumes the cloud is where the surface is; the threshold is a hardcoded
      constant, which means it is only meaningful for scenes normalised the way these are.
    * ``"stats_and_knn"`` / ``"both"`` -- keep only points that pass both.
    * ``"none"`` -- keep everything.

    Args:
        method: ``texture.filter_noisy_points``.
        surface_points: ``(M, 3)``.
        rgb: ``(M, 3)`` matching colours.
        model: For its point cloud, on the ``knn`` paths.
        args: The config tree, for the filter's parameters.

    Returns:
        The filtered ``(surface_points, rgb)``.
    """
    if method == "none":
        print("No noisy point filtering")
        return surface_points, rgb
    if method not in ("stats", "knn", "stats_and_knn", "both"):
        raise ValueError(f"Invalid noisy point filtering method: {method}")

    num_points = surface_points.shape[0]
    stats_mask = None
    knn_mask = None

    if method in ("stats", "stats_and_knn", "both"):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            surface_points.cpu().numpy().astype(np.float64))
        kwargs = dict(nb_neighbors=args.texture.filter_noisy_points_nb_neighbors,
                      std_ratio=args.texture.filter_noisy_points_std_ratio)
        try:
            _, keep = pcd.remove_statistical_outlier(print_progress=True, **kwargs)
        except TypeError:
            _, keep = pcd.remove_statistical_outlier(**kwargs)
        stats_mask = np.zeros(num_points, dtype=bool)
        stats_mask[np.array(keep)] = True

    if method in ("knn", "stats_and_knn", "both"):
        from pytorch3d.ops import knn_points

        model_points = model.points.detach()
        if args.texture.divide_by_coord_scale:
            model_points = model_points / model.coord_scale
        print("Sampled surface points before filtering:", surface_points.shape,
              surface_points.min(), surface_points.max())
        print("Model points for knn:", model_points.shape, model_points.min(), model_points.max())
        dists, _, _ = knn_points(surface_points.unsqueeze(0), model_points.unsqueeze(0), K=1,
                                 return_sorted=False)
        dists = dists.squeeze().detach().cpu().numpy()
        print("Distance stats: min {:.6f}, max {:.6f}, mean {:.6f}, median {:.6f}, std {:.6f}"
              .format(dists.min(), dists.max(), dists.mean(), np.median(dists), dists.std()))
        knn_mask = dists < 0.003

    if stats_mask is None:
        keep_mask = knn_mask
    elif knn_mask is None:
        keep_mask = stats_mask
    else:
        keep_mask = np.logical_and(stats_mask, knn_mask)

    print(f"Removed {num_points - int(keep_mask.sum())} points")
    keep_index = torch.from_numpy(np.nonzero(keep_mask)[0]).to(surface_points.device)
    surface_points = surface_points[keep_index]
    rgb = rgb[keep_index]
    print("After filtering, num fg pixels:", surface_points.shape[0])
    return surface_points, rgb


@torch.no_grad()
def extract_surface_points(model: Any, dataset: Any, args: Any, *, bbox_hull: Any = None,
                           use_gt_surface_points: bool = False) -> SurfacePointSet:
    """Render, mask, denoise and assemble the cloud the UV stage trains on.

    Args:
        model: A trained PAPR model, camera already set from ``dataset``.
        dataset: The training split.
        args: The resolved config tree (``eval``, ``texture`` and ``nuvo.sample`` are read).
        bbox_hull: A hull from :func:`uv.bbox.load_bounding_box`, or ``None`` to mask by alpha.
        use_gt_surface_points: Back-project the dataset's ground-truth depth instead of rendering.
            Requires ``dataset.load_gt_depth``; useful for isolating atlas quality from render
            quality, and off in the shipped config.

    Returns:
        A :class:`SurfacePointSet`.
    """
    divide = args.texture.divide_by_coord_scale
    erode = args.nuvo.sample.erode_fg_mask
    if erode:
        raise NotImplementedError(
            "nuvo.sample.erode_fg_mask is not implemented: this branch invoked a method that does "
            "not exist on the model, so it could never have run. Use "
            "nuvo.sample.bounding_box_mesh_path to tighten the foreground instead."
        )

    sampled_surface_points: list[Tensor] = []
    sampled_rgb: list[Tensor] = []

    if use_gt_surface_points:
        if getattr(dataset, "gt_surface_points", None) is None:
            raise ValueError(
                "nuvo.sample.use_gt_surface_points is enabled but the dataset has no ground-truth "
                "surface points; set dataset.load_gt_depth: true and dataset.gt_depth_dir"
            )
        print(f"Using ground truth surface points from dataset "
              f"({tuple(dataset.gt_surface_points.shape)})...")
        for view_idx in tqdm(range(dataset.num_imgs)):
            view_points = dataset.gt_surface_points[view_idx]
            if divide:
                view_points = view_points / model.coord_scale
            keep = _foreground_mask(dataset, view_idx, view_points, bbox_hull)
            sampled_surface_points.append(view_points[keep])
            sampled_rgb.append(dataset.images[view_idx][keep])
    else:
        tile_h = min(args.eval.max_height, dataset.H) if args.eval.max_height > 0 else dataset.H
        tile_w = min(args.eval.max_width, dataset.W) if args.eval.max_width > 0 else dataset.W
        print(f"Extracting surface points from {dataset.num_imgs} views...")
        for view_idx in tqdm(range(dataset.num_imgs)):
            view_points = render_view_surface_points(
                model, dataset, view_idx, tile_height=tile_h, tile_width=tile_w,
                divide_by_coord_scale=divide,
            )[0]
            keep = _foreground_mask(dataset, view_idx, view_points, bbox_hull)
            sampled_surface_points.append(view_points[keep])
            sampled_rgb.append(dataset.images[view_idx][keep])

    surface_points = torch.cat(sampled_surface_points, dim=0)
    rgb = torch.cat(sampled_rgb, dim=0)
    print("Num fg pixels:", surface_points.shape[0])
    print("Sampled surface points:", tuple(surface_points.shape))
    print("Sampled rgb:", tuple(rgb.shape))

    surface_points, rgb = filter_noisy_points(args.texture.filter_noisy_points, surface_points,
                                              rgb, model, args)

    num_original_surface_points = surface_points.shape[0]

    if args.texture.pcd_geom_loss_weight > 0:
        print("Appending point cloud to surface points for unified geometry loss...")
        points_in_pcd = model.points.detach()
        if divide:
            points_in_pcd = points_in_pcd / model.coord_scale
        surface_points = torch.cat([surface_points, points_in_pcd], dim=0)
        print(f"  Added {points_in_pcd.shape[0]} point cloud points. "
              f"Total points: {surface_points.shape[0]}")

    return SurfacePointSet(surface_points, rgb, num_original_surface_points)


def _foreground_mask(dataset: Any, view_idx: int, view_points: Tensor, bbox_hull: Any) -> Tensor:
    """``(H, W)`` boolean: which pixels of this view contribute a surface point."""
    if bbox_hull is not None:
        from uv.bbox import points_inside_bounding_box

        return points_inside_bounding_box(view_points, bbox_hull)
    return dataset.masks[view_idx].squeeze(-1) > 0.5


def farthest_point_sample(points: Tensor, num_samples: int, normals: Tensor | None = None,
                          normal_weight: float = 1.0) -> tuple[Tensor, Tensor]:
    """Greedy farthest-point sampling, in 3D or in 6D with normals appended.

    Picks well-separated anchors for K-means so its initial centroids are spread over the object
    instead of clumped where the points are dense. With ``normals``, two points on opposite sides of
    a thin structure are far apart even though they are close in space, which is what stops a chart
    from wrapping around an edge.

    The first anchor is drawn at random (from the tensor's device generator), so this is *not*
    deterministic across runs unless the seed is set.

    Args:
        points: ``(N, 3)``.
        num_samples: How many anchors. ``>= N`` returns everything.
        normals: ``(N, 3)`` unit normals for the 6D variant, or ``None``.
        normal_weight: How much a normal difference counts against a position difference.

    Returns:
        ``(indices, features[indices])``, where features is the 3D or 6D array actually used.
    """
    device = points.device
    n_points = points.shape[0]
    features = points if normals is None else torch.cat([points, normals * normal_weight], dim=1)
    if num_samples >= n_points:
        return torch.arange(n_points, device=device), features

    indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((n_points,), float("inf"), device=device)
    indices[0] = torch.randint(n_points, (1,), device=device)

    for i in range(1, num_samples):
        last = features[indices[i - 1]].unsqueeze(0)
        distances = torch.minimum(distances, torch.sum((features - last) ** 2, dim=1))
        indices[i] = torch.argmax(distances)

    return indices, features[indices]


def kmeans_lloyd(points: Tensor, anchors: Tensor, num_iters: int = 50) -> tuple[Tensor, Tensor]:
    """Plain Lloyd iteration on the GPU from fixed initial centroids.

    An empty cluster keeps its old centroid rather than being re-seeded, so a bad initialisation can
    leave a chart with no points. The sklearn paths do not have this problem.

    Args:
        points: ``(N, D)``.
        anchors: ``(K, D)`` initial centroids.
        num_iters: Iterations. No convergence check.

    Returns:
        ``(labels, centroids)``.
    """
    centroids = anchors.clone()
    n_clusters = anchors.shape[0]
    for _ in range(num_iters):
        labels = torch.argmin(torch.cdist(points, centroids), dim=1)
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            mask = labels == k
            new_centroids[k] = points[mask].mean(dim=0) if mask.sum() > 0 else centroids[k]
        centroids = new_centroids
    labels = torch.argmin(torch.cdist(points, centroids), dim=1)
    return labels, centroids


def estimate_normals(points: Tensor, radius: float | None = None, max_nn: int = 30) -> Tensor:
    """Per-point unit normals via open3d's local PCA, oriented toward the world origin.

    Orienting toward the origin rather than toward each camera is a real assumption: it is right for
    an object-centred capture and wrong for a scene the cameras sit inside. It only affects the
    *sign* of the normals, which for 6D clustering decides whether the two sides of a thin surface
    cluster together or apart.

    Args:
        points: ``(N, 3)``.
        radius: Neighbourhood radius, or ``None`` to estimate one as 5x the average spacing of a
            1000-point sample (which consumes the numpy RNG).
        max_nn: Neighbour cap per point.

    Returns:
        ``(N, 3)`` on ``points``' device. Degenerate normals become ``[0, 0, 1]``.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    device = points.device
    points_np = points.detach().cpu().numpy().astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    if radius is None:
        sample_size = min(1000, len(points_np))
        indices = np.random.choice(len(points_np), sample_size, replace=False)
        neighbour_dists, _ = cKDTree(points_np).query(points_np[indices], k=10)
        distances = neighbour_dists[:, 1:].ravel()
        avg_spacing = float(np.mean(distances)) if distances.size else 0.1
        radius = avg_spacing * 5
        print(f"  Estimated normal search radius: {radius:.4f} (avg spacing: {avg_spacing:.4f})")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    normals64 = np.asarray(pcd.normals).copy()
    reference = np.zeros(3, dtype=np.float64) - points_np
    zero = np.linalg.norm(normals64, axis=1) == 0.0
    facing_away = np.einsum("ij,ij->i", normals64, reference) < 0.0
    normals64[~zero & facing_away] *= -1.0
    if zero.any():
        ref_norm = np.linalg.norm(reference[zero], axis=1, keepdims=True)
        normals64[zero] = np.divide(reference[zero], ref_norm, out=np.zeros_like(reference[zero]),
                                    where=ref_norm > 0)

    normals = normals64.astype(np.float32)
    bad = np.isnan(normals).any(axis=1) | (np.linalg.norm(normals, axis=1) < 1e-6)
    if bad.any():
        print(f"  Warning: {bad.sum()} points have degenerate normals, using [0,0,1]")
        normals[bad] = [0, 0, 1]
    return torch.from_numpy(normals).to(device)


def knn_voting_cleanup(points: Tensor, labels: Tensor, k: int = 20,
                       num_iters: int = 1) -> Tensor:
    """Majority-vote each point's label among its ``k`` nearest neighbours, repeatedly.

    K-means in 6D produces speckle: single points whose normal happens to favour a neighbouring
    chart. Speckle is much worse for an atlas than for a segmentation, because every isolated point
    forces its chart's map to reach across the object. Voting erases it while leaving the boundaries
    where they are, and stops early once nothing changes.

    Args:
        points: ``(N, 3)``.
        labels: ``(N,)`` integer labels.
        k: Neighbours polled (the query asks for ``k + 1`` and includes the point itself, so it
            votes for its own current label).
        num_iters: Maximum passes.

    Returns:
        ``(N,)`` cleaned labels.
    """
    from scipy.spatial import cKDTree

    device = points.device
    points_np = points.detach().cpu().numpy().astype(np.float64)
    labels_np = labels.cpu().numpy().astype(np.int32)

    n_points = len(points_np)
    n_clusters = int(labels_np.max()) + 1
    _, neighbor_indices = cKDTree(points_np).query(points_np, k=k + 1)

    current = labels_np.copy()
    for iteration in range(num_iters):
        neighbor_labels = current[neighbor_indices]
        one_hot = np.zeros((n_points, k + 1, n_clusters), dtype=np.int32)
        np.put_along_axis(one_hot, neighbor_labels[:, :, None], 1, axis=2)
        new_labels = np.argmax(one_hot.sum(axis=1), axis=1).astype(np.int32)

        changed = int(np.sum(new_labels != current))
        print(f"  KNN voting iter {iteration + 1}: {changed} labels changed "
              f"({100 * changed / n_points:.2f}%)")
        current = new_labels
        if changed == 0:
            break

    return torch.from_numpy(current).long().to(device)


def _kmeans_sklearn(features: np.ndarray, n_clusters: int, init: Any, max_iters: int,
                    n_init: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """sklearn K-means, shared by the 3D and 6D paths."""
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iters, n_init=n_init,
                    random_state=random_state, verbose=0)
    labels = kmeans.fit_predict(features)
    print(f"  sklearn KMeans converged in {kmeans.n_iter_} iterations "
          f"(inertia: {kmeans.inertia_:.4f})")
    return labels, kmeans.cluster_centers_


def chart_assignment_clusters(points: Tensor, num_charts: int, train_conf: Any) -> Tensor:
    """Cluster surface points into ``num_charts`` groups, the warmup's target labels.

    Four routes, selected by ``chart_assignment_use_6d`` and ``chart_assignment_clustering_init``:
    3D or 6D features, each initialised by farthest-point sampling or by sklearn's k-means++. The
    editing config's choice -- 6D with k-means++ and 3 rounds of 100-neighbour voting -- clusters by
    position *and* orientation, which is what separates the front and back of a thin surface into
    different charts instead of one chart that wraps around it.

    Args:
        points: ``(N, 3)`` surface points in scene units.
        num_charts: Number of clusters.
        train_conf: The ``nuvo.train`` config node.

    Returns:
        ``(N,)`` integer labels in ``[0, num_charts)``.
    """
    if train_conf.get("chart_assignment_colored_pcd_path"):
        raise NotImplementedError(
            "nuvo.train.chart_assignment_colored_pcd_path (clustering by the colours of a "
            "hand-painted point cloud) is not ported; it is null in every shipped config. Use the "
            "K-means paths, or pre-train the assignment separately and pass it with --nuvo_ckpt."
        )

    kmeans_iters = train_conf.chart_assignment_warmup_kmeans_iters
    use_6d = train_conf.chart_assignment_use_6d
    clustering_init = train_conf.chart_assignment_clustering_init
    normal_weight = train_conf.chart_assignment_normal_weight
    n_init = train_conf.chart_assignment_sklearn_n_init
    random_state = train_conf.chart_assignment_sklearn_random_state

    normals = None
    if use_6d:
        print("\nStep 0: Estimating point normals...")
        normals = estimate_normals(points, radius=train_conf.chart_assignment_normal_radius,
                                   max_nn=train_conf.chart_assignment_normal_max_nn)
        print(f"  Estimated normals for {normals.shape[0]} points")

    if use_6d:
        features = torch.cat([points, normals * normal_weight], dim=1)
        if clustering_init == "fps":
            print("\nStep 1: Farthest Point Sampling to select anchor points (6D)...")
            _, anchors = farthest_point_sample(points, num_charts, normals=normals,
                                               normal_weight=normal_weight)
            print(f"Selected {num_charts} anchor points via 6D FPS")
            print("\nStep 2: 6D K-means clustering with FPS initialization...")
            labels_np, _ = _kmeans_sklearn(features.cpu().numpy(), num_charts,
                                           anchors.cpu().numpy(), kmeans_iters, 1, random_state)
        else:
            print("\nStep 1-2: 6D K-means clustering with auto-initialization...")
            labels_np, _ = _kmeans_sklearn(features.cpu().numpy(), num_charts, "k-means++",
                                           kmeans_iters, n_init, random_state)
        cluster_labels = torch.from_numpy(labels_np).long().to(points.device)
    elif clustering_init == "fps":
        print("\nStep 1: Farthest Point Sampling to select anchor points...")
        _, anchors = farthest_point_sample(points, num_charts)
        print(f"Selected {num_charts} anchor points via FPS")
        print(f"\nStep 2: K-means clustering with {kmeans_iters} iterations...")
        cluster_labels, _ = kmeans_lloyd(points, anchors, num_iters=kmeans_iters)
    else:
        print(f"\nStep 1-2: sklearn KMeans clustering (k={num_charts})...")
        labels_np, _ = _kmeans_sklearn(points.cpu().numpy(), num_charts, "k-means++",
                                       kmeans_iters, n_init, random_state)
        cluster_labels = torch.from_numpy(labels_np).long().to(points.device)

    voting_k = train_conf.chart_assignment_knn_voting_k
    voting_iters = train_conf.chart_assignment_knn_voting_iters
    if voting_k > 0 and voting_iters > 0:
        print(f"\nStep 2.5: KNN voting cleanup (k={voting_k}, iters={voting_iters})...")
        before = cluster_labels.clone()
        cluster_labels = knn_voting_cleanup(points, cluster_labels, k=voting_k,
                                            num_iters=voting_iters)
        changed = int((cluster_labels != before).sum().item())
        print(f"  Total labels changed: {changed} "
              f"({100 * changed / len(cluster_labels):.2f}%)")

    return cluster_labels


def _chart_colors(num_charts: int) -> np.ndarray:
    """``(num_charts, 3)`` RGB in ``[0, 1]``, one distinct hue per chart."""
    return np.array([colorsys.hsv_to_rgb(i / num_charts, 0.7, 0.8) for i in range(num_charts)])


def warmup_chart_assignment(nuvo_model: Any, nuvo_conf: Any, points: Tensor,
                            log_dir: str | Path | None = None) -> None:
    """Pre-train the chart assignment to reproduce a spatial clustering. Port of the stage
    script's ``warmup_chart_assignment_mlp`` (``:477``).

    Plain cross-entropy against K-means labels, with its own Adam and cosine schedule, before the
    joint optimisation starts. It changes *only* the chart assignment network; the coordinate maps
    are untouched, so what it buys is a sensible starting partition of the object rather than the
    symmetric one a random init gives.

    Does nothing unless ``nuvo.train.chart_assignment_warmup`` is set (it is false in the shipped
    config). Returns after training; the caller keeps the model.

    Args:
        nuvo_model: The atlas whose ``chart_assignment_mlp`` is trained.
        nuvo_conf: The ``nuvo`` config node.
        points: ``(N, 3)`` surface points in the frame the chart networks see -- i.e. already
            divided by ``coord_scale`` when ``texture.divide_by_coord_scale`` is set.
        log_dir: Where to write the target clustering as a colour-coded PLY, or ``None`` to skip.
            This keeps the point cloud, which is the artifact worth inspecting.
    """
    train_conf = nuvo_conf.train
    if not train_conf.get("chart_assignment_warmup", False):
        print("Chart assignment warmup is disabled, skipping...")
        return

    print("=" * 60)
    print("Starting Chart Assignment MLP Warmup Training")
    print("=" * 60)

    num_charts = nuvo_conf.model.num_charts
    warmup_iters = train_conf.chart_assignment_warmup_iters
    batch_size = train_conf.chart_assignment_warmup_batch_size
    warmup_lr = train_conf.chart_assignment_warmup_lr
    log_interval = train_conf.chart_assignment_warmup_log_interval

    print(f"Number of charts: {num_charts}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {warmup_lr}")
    print(f"Total surface points: {points.shape[0]}")

    cluster_labels = chart_assignment_clusters(points, num_charts, train_conf)

    print("\nCluster statistics:")
    for i in range(num_charts):
        count = int((cluster_labels == i).sum().item())
        print(f"  Chart {i}: {count} points ({100 * count / points.shape[0]:.1f}%)")

    if log_dir is not None:
        warmup_dir = Path(log_dir) / "warmup_chart_assignment"
        warmup_dir.mkdir(parents=True, exist_ok=True)
        colors = _chart_colors(num_charts)[cluster_labels.cpu().numpy()]
        write_ply_points(warmup_dir / "target_clusters.ply", points.detach().cpu().numpy(),
                         colors=colors)
        print(f"Saved target clustering to {warmup_dir / 'target_clusters.ply'}")

    print(f"\nStep 3: Training chart assignment MLP for {warmup_iters} iterations...")
    optimizer = torch.optim.Adam(nuvo_model.chart_assignment_mlp.parameters(), lr=warmup_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_iters)
    loss_fn = nn.CrossEntropyLoss()
    num_points = points.shape[0]

    for step in tqdm(range(warmup_iters), desc="Chart Assignment Warmup"):
        batch_indices = torch.randint(0, num_points, (batch_size,), device=points.device)
        batch_points = points[batch_indices]
        batch_labels = cluster_labels[batch_indices]

        optimizer.zero_grad()
        logits = nuvo_model.chart_assignment_mlp(batch_points, return_logits=True)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0 or step == warmup_iters - 1:
            with torch.no_grad():
                accuracy = (torch.argmax(logits, dim=1) == batch_labels).float().mean().item()
            print(f"  Step {step}: loss={loss.item():.4f}, accuracy={accuracy * 100:.1f}%, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    print("\nFinal evaluation on all points...")
    with torch.no_grad():
        predictions = []
        for start in range(0, num_points, batch_size):
            batch = points[start:min(start + batch_size, num_points)]
            predictions.append(torch.argmax(
                nuvo_model.chart_assignment_mlp(batch, return_logits=True), dim=1))
        predicted = torch.cat(predictions)
        accuracy = (predicted == cluster_labels).float().mean().item()
        print(f"Final accuracy on all points: {accuracy * 100:.1f}%")
        print("\nPredicted cluster statistics:")
        for i in range(num_charts):
            count = int((predicted == i).sum().item())
            print(f"  Chart {i}: {count} points ({100 * count / num_points:.1f}%)")

    print("=" * 60)
    print("Chart Assignment MLP Warmup Training Complete!")
    print("=" * 60)

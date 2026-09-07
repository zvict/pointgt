from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from models.activations import make_activation
from utils.geometry import normalize_vector, project_point_onto_ray, rectify_points

try:
    import tinycudann as tcnn
except ImportError:  # pragma: no cover - depends on the machine, not on the code
    tcnn = None

logger = logging.getLogger(__name__)

__all__ = [
    "Encoding",
    "KQVDims",
    "K_FEATURE_DIMS",
    "PositionalEncoding",
    "Q_FEATURE_DIMS",
    "RayPointGeometry",
    "V_FEATURE_DIMS",
    "as_tcnn_config",
    "get_tcnn_init_weights",
    "key_features",
    "kqv_dims",
    "positional_encoding",
    "query_features",
    "ray_point_geometry",
    "value_features",
]


#: Input width of the key network, per ``models.attn.k_type``.
#: 1 = ``[point, along_ray, to_ray]``; 13 = the point in the ray's own rectified frame.
K_FEATURE_DIMS: dict[int, int] = {1: 3 + 3 + 3, 13: 3}

#: Input width of the query network, per ``models.attn.q_type``.
#: 1 = the ray direction; 5 = a constant one-vector (the query carries no ray information).
Q_FEATURE_DIMS: dict[int, int] = {1: 3, 5: 3}

#: Input width of the value network, per ``models.attn.v_type``.
#: 1 = ``[along_ray, to_ray]``; 2 = the same with the along-ray part normalised to unit length.
V_FEATURE_DIMS: dict[int, int] = {1: 3 + 3, 2: 3 + 3}


@dataclass(frozen=True)
class KQVDims:
    """Input widths of the three feature networks, before any positional encoding."""

    k: int
    q: int
    v: int


def _lookup_dim(table: Mapping[int, int], feat_type: int, role: str) -> int:
    try:
        return table[int(feat_type)]
    except (KeyError, TypeError, ValueError):
        supported = ", ".join(str(t) for t in sorted(table))
        raise ValueError(
            f"unsupported models.attn.{role}_type {feat_type!r}; this release implements "
            f"{{{supported}}}. Other variants are not reachable from any shipped "
            "config and were not ported."
        ) from None


def kqv_dims(k_type: int, q_type: int, v_type: int) -> KQVDims:
    """Widths the key/query/value networks must be built for.

    This is the release's ``get_kqv_dim``. Returning the widths explicitly, rather than writing
    them as a side effect on the module, means the widths and the feature builders cannot silently
    disagree after a config edit; it makes the attention module state the contract at construction
    time.

    The same tables serve ``prop_k_type`` / ``prop_q_type``, which default to 1 and index the same
    key/query feature space.
    """
    return KQVDims(
        k=_lookup_dim(K_FEATURE_DIMS, k_type, "k"),
        q=_lookup_dim(Q_FEATURE_DIMS, q_type, "q"),
        v=_lookup_dim(V_FEATURE_DIMS, v_type, "v"),
    )


@dataclass(frozen=True)
class RayPointGeometry:
    """The orthogonal split of ``point - ray_origin`` every attention feature is cut from."""

    #: ``(N, H, W, K, 1)`` unsigned distance along the ray to the foot of the perpendicular.
    pd: Tensor
    #: ``(N, H, W, K, 1)`` distance from the point to the infinite ray.
    d2r: Tensor
    #: ``(N, H, W, K, 3)`` along-ray component, ``t * direction``.
    vec_pd: Tensor
    #: ``(N, H, W, K, 3)`` across-ray component, ``point - foot``.
    vec_d2r: Tensor


def ray_point_geometry(rays_o: Tensor, rays_d: Tensor, points: Tensor) -> RayPointGeometry:
    """Decompose each candidate point against its ray.

    Args:
        rays_o: ``(N, 3)`` one origin per view, or ``(N, H, W, 3)`` one per ray.
        rays_d: ``(N, H, W, 3)`` ray directions, **already unit norm**. They are not renormalised
            here, because renormalising an already-unit vector perturbs the last bits of every
            attention feature.
        points: ``(N, H, W, K, 3)`` the ``K`` candidate points selected for each ray.

    Returns:
        A :class:`RayPointGeometry`. The caller keeps all four tensors because the key, query and
        value builders each need a different subset and recomputing is the render loop's hot path.
    """
    n_views, _, _, _ = rays_d.shape

    rays_d = rays_d.unsqueeze(-2)
    if rays_o.ndim == 2:
        rays_o = rays_o.reshape(n_views, 1, 1, 1, 3)
    else:
        rays_o = rays_o.unsqueeze(-2)

    projection = project_point_onto_ray(points, rays_o, rays_d, normalize=False)
    return RayPointGeometry(
        pd=projection.projected_distance,
        d2r=projection.distance_to_ray,
        vec_pd=projection.parallel,
        vec_d2r=projection.perpendicular,
    )


def key_features(
    feat_type: int,
    *,
    rays_o: Tensor,
    rays_d: Tensor,
    selected_points: Tensor,
    vec_pd: Tensor,
    vec_d2r: Tensor,
    pd: Tensor,
    d2r: Tensor | None = None,
    z: Tensor | None = None,
) -> Tensor:
    """Build the key network's input, ``(N, H, W, K, K_FEATURE_DIMS[feat_type])``.

    Args:
        feat_type: ``models.attn.k_type``. 1 or 13.
        rays_o: ``(N, 3)``. Type 13 reshapes it to ``(N, 1, 3)``, so a per-ray origin is rejected.
        rays_d: ``(N, H, W, 3)``, unit norm.
        selected_points: ``(N, H, W, K, 3)`` world-space candidate points.
        vec_pd, vec_d2r, pd: from :func:`ray_point_geometry`.
        d2r, z: accepted for signature compatibility; unused by types 1 and 13.

    Raises:
        ValueError: for any other ``feat_type``.
    """
    del d2r, z
    n_views, height, width, num_candidates, _ = vec_pd.shape
    feat_type = int(feat_type)

    if feat_type == 1:
        return torch.cat([selected_points.detach(), vec_pd, vec_d2r], dim=-1)

    if feat_type == 13:
        rays = height * width
        rectified = rectify_points(
            selected_points.reshape(n_views, rays, num_candidates, 3),
            rays_o.reshape(n_views, 1, 3).expand(-1, rays, -1),
            rays_d.reshape(n_views, rays, 3),
            translate=True,
            ts=pd.reshape(n_views, rays, num_candidates),
        )
        return rectified.points.reshape(n_views, height, width, num_candidates, 3)

    raise ValueError(
        f"unsupported models.attn.k_type {feat_type}; this release implements {{1, 13}}. The other "
        "key variants are not reachable from any shipped config."
    )


def query_features(
    feat_type: int,
    *,
    rays_o: Tensor | None = None,
    rays_d: Tensor,
    selected_points: Tensor | None = None,
    vec_pd: Tensor | None = None,
    vec_d2r: Tensor | None = None,
    pd: Tensor | None = None,
    d2r: Tensor | None = None,
    z: Tensor | None = None,
) -> Tensor:
    """Build the query network's input, ``(N, H, W, Q_FEATURE_DIMS[feat_type])``.

    Note the missing ``K`` axis: there is one query per ray, scored against all ``K`` keys.

    Type 5 returns ``ones_like(rays_d)``, i.e. a constant. That is not a placeholder -- it is what
    every released config uses. The query then contributes only through the (learned) query network
    and its projection, so the attention score depends on the key alone, and the ray's direction
    enters the model solely through the rectified key frame. Ablating it back to type 1 changes the
    objective, so it must stay selectable.

    Args:
        feat_type: ``models.attn.q_type``. 1 or 5.
        rays_d: ``(N, H, W, 3)``, unit norm.
        rays_o, selected_points, vec_pd, vec_d2r, pd, d2r, z: accepted for signature
            compatibility; unused by types 1 and 5.

    Raises:
        ValueError: for any other ``feat_type``.
    """
    del rays_o, selected_points, vec_pd, vec_d2r, pd, d2r, z
    feat_type = int(feat_type)

    if feat_type == 1:
        return rays_d
    if feat_type == 5:
        return torch.ones_like(rays_d)

    raise ValueError(
        f"unsupported models.attn.q_type {feat_type}; this release implements {{1, 5}}. The other "
        "query variants are not reachable from any shipped config."
    )


def value_features(
    feat_type: int,
    *,
    rays_o: Tensor | None = None,
    rays_d: Tensor | None = None,
    points: Tensor | None = None,
    select_k_ind: Tensor | None = None,
    selected_points: Tensor | None = None,
    vec_pd: Tensor,
    vec_d2r: Tensor,
    pd: Tensor | None = None,
    d2r: Tensor | None = None,
    z: Tensor | None = None,
) -> Tensor:
    """Build the value network's geometric input, ``(..., V_FEATURE_DIMS[feat_type])``.

    The per-point learned feature vector is concatenated onto this by the attention module *after*
    the value encoder runs, which is why only the geometry appears here.

    Both shipped types are the same pair of vectors; type 2 normalises the along-ray part, which
    discards the point's depth along the ray and leaves only its sign-carrying direction. Type 1
    keeps the raw magnitude. Neither is a strict refinement of the other and the released configs
    use both.

    The trailing shape is whatever ``vec_pd`` and ``vec_d2r`` carry, so this works both on the
    ``(N, H, W, K, 3)`` render path and on a flattened ``(N*H*W*K, 3)`` one.

    Args:
        feat_type: ``models.attn.v_type``. 1 or 2.
        vec_pd, vec_d2r: from :func:`ray_point_geometry`.
        rays_o, rays_d, points, select_k_ind, selected_points, pd, d2r, z: accepted for signature
            compatibility; unused by types 1 and 2.

    Raises:
        ValueError: for any other ``feat_type``.
    """
    del rays_o, rays_d, points, select_k_ind, selected_points, pd, d2r, z
    feat_type = int(feat_type)

    if feat_type == 1:
        return torch.cat([vec_pd, vec_d2r], dim=-1)
    if feat_type == 2:
        return torch.cat([normalize_vector(vec_pd), vec_d2r], dim=-1)

    raise ValueError(
        f"unsupported models.attn.v_type {feat_type}; this release implements {{1, 2}}. The other "
        "value variants are not reachable from any shipped config."
    )


def positional_encoding(
    x: Tensor,
    n_frequencies: int,
    factor: float = 2.0,
    without_self: bool = False,
    mult_factor: float = 1.0,
    stop_gradient: bool = False,
) -> Tensor:
    """NeRF-style axis-aligned Fourier features, the pure-torch alternative to tinycudann's.

    The output interleaves by *input channel* rather than by frequency: channel ``i``'s raw value
    and all of its sin/cos terms are adjacent. That ordering matters only in that a checkpoint
    trained with it cannot be loaded into a differently-ordered encoder.

    Unlike tinycudann's ``Frequency`` encoding this applies **no** factor of pi -- see
    :class:`Encoding`, where the two are reconciled.
    """
    rets: list[Tensor] = [] if without_self else [x]
    for i in range(n_frequencies):
        for fn in (torch.sin, torch.cos):
            code = fn(factor**i * x * mult_factor)
            rets.append(code.detach() if stop_gradient else code)
    return torch.flatten(torch.stack(rets, -1), start_dim=-2, end_dim=-1)


class PositionalEncoding(nn.Module):
    """Stateless module wrapper around :func:`positional_encoding`."""

    def __init__(self, factor: float = 2.0, mult_factor: float = 1.0) -> None:
        super().__init__()
        self.factor = float(factor)
        self.mult_factor = float(mult_factor)

    def forward(
        self,
        x: Tensor,
        n_frequencies: int,
        without_self: bool = False,
        stop_gradient: bool = False,
    ) -> Tensor:
        return positional_encoding(
            x, n_frequencies, self.factor, without_self, self.mult_factor, stop_gradient
        )

    def extra_repr(self) -> str:
        return f"factor={self.factor}, mult_factor={self.mult_factor}"


def as_tcnn_config(config: Any) -> dict[str, Any]:
    """Unwrap a config block into the plain ``dict`` tinycudann's pybind layer requires.

    ``tcnn.Encoding`` and ``tcnn.Network`` take their config as a nlohmann ``json``, and pybind
    converts only a real ``dict`` -- handed a ``ConfigNode`` (or any other Mapping) it raises
    ``TypeError: incompatible function arguments``. Every tinycudann construction in this release
    must route its config through here.

    Extra keys the encoding does not know (``with_self``, ``stop_gradient``, ``pe_scale``) are kept:
    tinycudann ignores them, and stripping them would diverge from the config that produced the
    released checkpoints.
    """
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if isinstance(config, Mapping):
        return {k: as_tcnn_config(v) if isinstance(v, Mapping) else v for k, v in config.items()}
    raise TypeError(f"tinycudann config must be a mapping, got {type(config).__name__}")


class Encoding(nn.Module):
    """The input encoder in front of a key/query/value network.

    Wraps either a tinycudann encoding or :class:`PositionalEncoding`, and adds the raw input back
    on the Python side when ``with_self`` is set -- tinycudann's ``Frequency`` encoding does not
    emit the identity term itself, so the concatenation cannot move into the config.

    Three preserved behaviours:

    * **The pi rescale.** tinycudann's ``Frequency`` multiplies its input by pi before taking
      sin/cos; NeRF does not. The input is divided by pi to cancel that; removing the division
      stopped the attention from recovering after the first prune. The division is written as
      ``x * pe_scale / pi`` -- that grouping, not ``x * (pe_scale / pi)``, is what the checkpoints
      were trained with.
    * **``stop_gradient`` only bites when ``with_self`` is set.** In the tinycudann branch the
      detach is applied inside the ``with_self`` concatenation, so a config with
      ``stop_gradient: true, with_self: false`` silently keeps the gradient. The pure-torch branch
      does honour it in both cases.
    * **``n_frequencies: 0`` requires ``with_self``**, otherwise the encoder would emit width zero.
      This is enforced as an explicit error (not an ``assert``) so it survives ``python -O``.

    Args:
        input_dim: Width of the tensor this encodes.
        config: One of the ``encode_*_config`` blocks -- ``otype``, ``n_frequencies``,
            ``with_self``, ``stop_gradient``, ``pe_scale``. Passed to tinycudann whole, extra keys
            included.
        use_tcnn_encoder: Use tinycudann (``models.attn.use_tcnn_encoder``). False selects the
            pure-torch encoder, whose output ordering and scaling differ -- the two are not
            checkpoint-compatible with each other.
    """

    def __init__(self, input_dim: int, config: Any, use_tcnn_encoder: bool = True) -> None:
        super().__init__()
        self.config = config
        self.use_tcnn_encoder = bool(use_tcnn_encoder)
        self.input_dim = int(input_dim)

        n_frequencies = int(config.n_frequencies)
        if n_frequencies > 0:
            if self.use_tcnn_encoder:
                if tcnn is None:
                    raise ImportError(
                        "models.attn.use_tcnn_encoder is true but tinycudann is not importable. "
                        "Install tiny-cuda-nn, or set use_tcnn_encoder: false and the network "
                        "otype to 'MLP' for the pure-torch path (different numerics, not "
                        "checkpoint-compatible)."
                    )
                self.encoder = tcnn.Encoding(
                    self.input_dim, as_tcnn_config(config), dtype=torch.float32
                )
                encoded_dims = self.encoder.n_output_dims
            else:
                self.encoder = PositionalEncoding(mult_factor=config.pe_scale)
                encoded_dims = self.input_dim * 2 * n_frequencies
            self.n_output_dims = encoded_dims + self.input_dim if config.with_self else encoded_dims
        else:
            if not config.with_self:
                raise ValueError(
                    "an encoding with n_frequencies == 0 and with_self == false would produce a "
                    "zero-width output; set with_self: true or give it frequencies"
                )
            self.encoder = None
            self.n_output_dims = self.input_dim

        logger.debug(
            "Encoding(input_dim=%d, tcnn=%s) -> %d dims",
            self.input_dim,
            self.use_tcnn_encoder,
            self.n_output_dims,
        )

    def forward(self, x: Tensor) -> Tensor:
        config = self.config
        if int(config.n_frequencies) <= 0:
            return x

        if self.use_tcnn_encoder:
            code = self.encoder(x * config.pe_scale / math.pi)
            if config.with_self:
                code = torch.cat([x, code.detach() if config.stop_gradient else code], dim=-1)
            return code

        return self.encoder(
            x,
            int(config.n_frequencies),
            without_self=not config.with_self,
            stop_gradient=config.stop_gradient,
        )

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, n_output_dims={self.n_output_dims}"


def get_tcnn_init_weights(input_dims: int, output_dims: int, network_config: Any) -> Tensor:
    """Xavier-initialise a fused tinycudann network by building its torch twin and flattening it.

    ``tcnn.Network`` exposes its parameters as one flat half-precision vector, and its default
    initialisation is not Xavier. This function builds an equivalent bias-free
    :class:`torch.nn.Sequential` on the GPU, applies ``xavier_uniform_``, and copies the flattened
    weights across to reproduce that byte layout.

    **The 16-multiple padding is the whole point.** tinycudann pads a layer's input and output
    widths up to a multiple of 16, and stores the padded matrices, so the flat vector is longer than
    the logical weight count. The input layer is padded along its *columns* (fan-in) and the output
    layer along its *rows* (fan-out), both with zeros. Get either wrong and the copy either raises
    on a size mismatch or, worse, silently shifts every subsequent layer's weights.

    Note that the call sites pass ``output_dim_k - 1`` for the key network while passing
    ``output_dim_q`` and ``output_dim_v`` unmodified for the others. With the shipped
    ``output_dim_k: 128`` that off-by-one changes nothing about the *shape* -- 127 pads back to 128
    -- but it does change the initialisation: the last output channel starts at exactly zero and
    the rest are drawn with Xavier's fan-out of 127. That is how the released checkpoints were
    initialised; a port must keep the ``- 1``.

    Args:
        input_dims: Logical fan-in of the first layer (the encoder's output width).
        output_dims: Logical fan-out of the last layer.
        network_config: A ``network_*_config`` block: ``n_hidden_layers``, ``n_neurons``,
            ``activation``, ``output_activation``.

    Returns:
        A flat ``float16`` tensor sized for ``tcnn.Network(...).params``, on the CUDA device.

    Raises:
        NotImplementedError: if ``n_hidden_layers`` is 0. tinycudann's single-matrix layout for a
            depth-0 network is not the concatenation this builds, and no shipped config asks for it.
    """
    n_hidden_layers = int(network_config.n_hidden_layers)
    n_neurons = int(network_config.n_neurons)
    if n_hidden_layers <= 0:
        raise NotImplementedError("No hidden layers in the network")

    modules: list[nn.Module] = [
        nn.Linear(input_dims, n_neurons, bias=False),
        make_activation(network_config.activation),
    ]
    for _ in range(n_hidden_layers - 1):
        modules.append(nn.Linear(n_neurons, n_neurons, bias=False))
        modules.append(make_activation(network_config.activation))
    modules.append(nn.Linear(n_neurons, output_dims, bias=False))
    if network_config.output_activation != "None":
        modules.append(make_activation(network_config.output_activation))

    model = nn.Sequential(*modules).cuda()
    model.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

    linear_layers = [m for m in model if isinstance(m, nn.Linear)]
    input_layer_weights = linear_layers[0].weight.data
    output_layer_weights = linear_layers[-1].weight.data
    if input_dims % 16 != 0:
        input_layer_weights = nn.functional.pad(
            input_layer_weights, (0, 16 - (input_dims % 16)), value=0
        )
    if output_dims % 16 != 0:
        output_layer_weights = nn.functional.pad(
            output_layer_weights, (0, 0, 0, 16 - (output_dims % 16)), value=0
        )

    weights = (
        [input_layer_weights]
        + [m.weight.data for m in linear_layers[1:-1]]
        + [output_layer_weights]
    )
    logger.debug(
        "tcnn init %d -> %d: %s",
        input_dims,
        output_dims,
        [tuple(w.shape) for w in weights],
    )
    return torch.cat([w.flatten() for w in weights]).half()

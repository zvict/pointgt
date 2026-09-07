from __future__ import annotations

import colorsys
import math
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.activations import make_activation
from models.features import PositionalEncoding, as_tcnn_config, get_tcnn_init_weights

try:
    import tinycudann as tcnn
except ImportError:  # pragma: no cover - exercised only on a machine without the extension
    tcnn = None

__all__ = [
    "BiasLayer",
    "ChartAssignmentMLP",
    "ChartOutputs",
    "FirstLayerWithZeroInit",
    "MappingMLP",
    "Nuvo",
    "NuvoEncoding",
    "OPTIMIZER_GROUPS",
    "bilinear_interpolation",
    "checkerboard_textures",
]

#: Every optimizer group name ``nuvo.train.fix_keys`` may name. A group not built by the current
#: configuration (no texture map, say) is simply absent from ``model.optimizers``.
OPTIMIZER_GROUPS: tuple[str, ...] = (
    "chart_assignment",
    "texture_coordinate",
    "surface_coordinate",
    "sigma",
    "texture_map",
    "texture_mlp",
    "texture_map_bkg_feat",
    "additional_texture_map",
    "additional_texture_map_bkg_feat",
)


def bilinear_interpolation(grid: Tensor, uvs: Tensor, interpolation_mode: str = "bilinear",
                           padding_mode: str = "zeros") -> Tensor:
    """Sample a chart's texture image at continuous UV coordinates.

    UVs are in ``[0, 1]`` and are mapped to ``grid_sample``'s
    ``[-1, 1]`` with ``align_corners=True``, so ``uv = 0`` reads the *centre* of the corner texel
    rather than its outer edge -- the convention the texture maps were trained under, and the one
    :meth:`Nuvo.get_rgb_for_texture_map`'s resampling matches.

    UVs outside ``[0, 1]`` are handled by ``padding_mode`` (``"zeros"`` everywhere), which is why a
    chart that has not yet learned to keep its UVs in range reads black rather than wrapping around.
    That is also what makes ``uv.losses.uv_range_loss`` necessary.

    Args:
        grid: ``(H, W, C)`` texture image for one chart.
        uvs: ``(N, 2)`` coordinates in ``[0, 1]``; ``uvs[:, 0]`` indexes the width axis.
        interpolation_mode: ``grid_sample`` mode (``texture.interpolation_mode``).
        padding_mode: ``grid_sample`` padding (``texture.padding_mode``).

    Returns:
        ``(N, C)``.
    """
    grid_tensor = grid.permute(2, 0, 1).unsqueeze(0)
    uvs_tensor = (uvs * 2 - 1).unsqueeze(0).unsqueeze(0)
    interpolated = F.grid_sample(grid_tensor, uvs_tensor, align_corners=True,
                                 mode=interpolation_mode, padding_mode=padding_mode)
    return interpolated.permute(3, 1, 2, 0).squeeze(-1).squeeze(-1)


def checkerboard_textures(size: tuple[int, int], num_squares: int,
                          dark_colors: list[np.ndarray]) -> list[np.ndarray]:
    """One checkerboard per chart, in a distinct hue, for visualising the parameterisation.

    Rendering these instead of the learned texture shows the UV
    layout directly: square cells mean a near-isometric chart, sheared or bunched cells mean
    distortion, and a hue change across a surface marks a chart boundary.

    ``colorsys`` replaces ``matplotlib.colors.hsv_to_rgb`` -- same standard conversion, no plotting
    dependency.

    Args:
        size: ``(width, height)`` of each texture.
        num_squares: Cells along one side. ``width // num_squares`` truncates, so a size that is not
            a multiple leaves the last few columns/rows black.
        dark_colors: One ``(3,)`` RGB array in ``[0, 255]`` per chart.

    Returns:
        A list of ``(height, width, 3)`` ``uint8`` arrays.
    """
    textures = []
    width, height = size
    square_size = width // num_squares

    for dark_color in dark_colors:
        light_color = np.clip(dark_color * 1.2, 0, 255).astype(np.uint8)
        texture = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(num_squares):
            for j in range(num_squares):
                color = dark_color if (i + j) % 2 == 0 else light_color
                texture[i * square_size:(i + 1) * square_size,
                        j * square_size:(j + 1) * square_size] = color
        textures.append(texture)

    return textures


class NuvoEncoding(nn.Module):
    """Fourier features in front of a chart network.

    A near-copy of :class:`models.features.Encoding` that differs in exactly one numerically
    material way: the tinycudann branch here passes ``x * pe_scale`` where the PAPR encoder passes
    ``x * pe_scale / pi``. Both are as their trees left them. Keeping the Nuvo variant separate is
    what lets a Nuvo checkpoint and a PAPR checkpoint coexist in one process.

    The non-tinycudann branch is shared (:class:`models.features.PositionalEncoding`) and is
    identical in both trees.

    Args:
        input_dim: Width of the tensor to encode (3 for positions, 2 for UVs).
        config: ``otype`` / ``n_frequencies`` / ``with_self`` / ``pe_scale``. A plain mapping.
        use_tcnn_encoder: Use tinycudann's fused encoding.
    """

    def __init__(self, input_dim: int, config: dict[str, Any],
                 use_tcnn_encoder: bool = True) -> None:
        super().__init__()
        self.config = dict(config)
        self.use_tcnn_encoder = bool(use_tcnn_encoder)
        self.input_dim = int(input_dim)

        n_frequencies = int(self.config["n_frequencies"])
        if n_frequencies > 0:
            if self.use_tcnn_encoder:
                if tcnn is None:
                    raise ImportError(
                        "nuvo.model.use_tcnn is true but tinycudann is not importable. Install "
                        "tiny-cuda-nn, or set nuvo.model.use_tcnn: false for the pure-torch path "
                        "(different numerics, not checkpoint-compatible)."
                    )
                self.encoder = tcnn.Encoding(self.input_dim, as_tcnn_config(self.config),
                                             dtype=torch.float32)
                encoded_dims = self.encoder.n_output_dims
            else:
                self.encoder = PositionalEncoding(mult_factor=self.config["pe_scale"])
                encoded_dims = self.input_dim * 2 * n_frequencies
            self.n_output_dims = (encoded_dims + self.input_dim
                                  if self.config["with_self"] else encoded_dims)
        else:
            if not self.config["with_self"]:
                raise ValueError(
                    "an encoding with n_frequencies == 0 and with_self == false would produce a "
                    "zero-width output; set with_self: true or give it frequencies"
                )
            self.encoder = None
            self.n_output_dims = self.input_dim

    def forward(self, x: Tensor) -> Tensor:
        config = self.config
        if int(config["n_frequencies"]) <= 0:
            return x

        if not self.use_tcnn_encoder:
            return self.encoder(x, int(config["n_frequencies"]),
                                without_self=not config["with_self"])

        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        code = self.encoder(flat * config["pe_scale"])
        if config["with_self"]:
            code = torch.cat([flat, code], dim=-1)
        return code.reshape(*lead, code.shape[-1])

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, n_output_dims={self.n_output_dims}"


class FirstLayerWithZeroInit(nn.Module):
    """Linear-plus-zeroed-Fourier input layer: starts affine, learns to be more.

    ``W . x`` is Xavier-initialised as usual, but the branch that consumes the positional encoding
    starts at exactly zero in both weight and bias. The network is therefore *exactly* an affine
    function of its input at step 0 no matter how many frequencies the encoding has, and the
    high-frequency capacity is switched on only as fast as the gradient asks for it. For a chart
    map, that is the difference between charts that start flat and grow detail, and charts that
    start crumpled and never recover.

    Args:
        input_dim: Input width.
        output_dim: Hidden width of the network this feeds.
        use_tcnn: Build the encoding with tinycudann.
        degree: Number of Fourier octaves (``*_pe_degree``). Zero gives an identity encoding.
        scale_by_pi: Divide the encoding's input by pi (``nuvo.model.scale_by_pi``, true in every
            config). Unlike the PAPR encoder's pi division this cancels nothing -- it is simply the
            input scale the checkpoints were trained at.
    """

    def __init__(self, input_dim: int, output_dim: int, use_tcnn: bool = False, degree: int = 0,
                 scale_by_pi: bool = False) -> None:
        super().__init__()
        self.scale_by_pi = bool(scale_by_pi)
        encoding_config = {
            "otype": "Frequency",
            "n_frequencies": degree,
            "with_self": True,
            "pe_scale": 1.0,
        }
        self.encoding = NuvoEncoding(input_dim, encoding_config, use_tcnn_encoder=use_tcnn)
        self.layer = nn.Linear(input_dim, output_dim)
        nn.init.xavier_normal_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)
        self.zero_init_layer = nn.Linear(self.encoding.n_output_dims, output_dim)
        nn.init.zeros_(self.zero_init_layer.weight)
        nn.init.zeros_(self.zero_init_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        encoded_x = self.encoding(x / math.pi if self.scale_by_pi else x).float()
        return (self.layer(x) + self.zero_init_layer(encoded_x)).float()


class BiasLayer(nn.Module):
    """A learned additive output bias, drawn from ``N(init_bias, init_bias_std)``.

    Its whole job is to break symmetry between charts. With ``s_bias_std: 1.0`` the surface-
    coordinate networks start centred on *different* points in space, so the charts begin by
    covering different parts of the object instead of collapsing onto one another. Using a nonzero
    value here is well earned: at 0.0 the charts start identical and the cycle losses have no
    gradient that separates them.

    The draw happens even when ``init_bias_std`` is 0 (giving a constant), which keeps the RNG
    stream aligned with configs that do use a spread.
    """

    def __init__(self, input_dim: int, init_bias: float = 0.0, init_bias_std: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.normal(init_bias, init_bias_std, size=(input_dim,)),
                                 requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.bias


def _tcnn_network(hidden_dim: int, output_dim: int, act: str, num_layers: int,
                  init_output_dim: int | None) -> nn.Module:
    """A fused tinycudann MLP body, Xavier-initialised through its torch twin.

    ``FullyFusedMLP`` only supports widths of 16/32/64/128, so anything else falls back to
    ``CutlassMLP`` -- the selection rule is kept because it decides which kernel a config
    lands on and therefore its numerics.

    Args:
        hidden_dim: Width of every hidden layer, and of the input (the front end's output).
        output_dim: Logical output width.
        act: Hidden activation name.
        num_layers: Total layers counting the front end, so the body has ``num_layers - 1``.
        init_output_dim: Fan-out to Xavier-initialise for, or ``None`` to leave tinycudann's own
            initialisation in place. Call sites pass ``output_dim - 1`` here (see
            :class:`MappingMLP`); ``None`` reproduces the unshared branch, which never initialised.
    """
    network_type = "FullyFusedMLP" if hidden_dim in (16, 32, 64, 128) else "CutlassMLP"
    network_config = {
        "otype": network_type,
        "activation": act,
        "output_activation": "None",
        "n_neurons": hidden_dim,
        "n_hidden_layers": num_layers - 1,
    }
    if tcnn is None:
        raise ImportError(
            "nuvo.model.use_tcnn is true but tinycudann is not importable. Install tiny-cuda-nn, "
            "or set nuvo.model.use_tcnn: false."
        )
    network = tcnn.Network(hidden_dim, output_dim, network_config)
    if init_output_dim is not None:
        from utils.runtime import to_config_node

        network.params.data[...] = get_tcnn_init_weights(
            hidden_dim, init_output_dim, to_config_node(network_config)
        )
    return network


def _torch_network(hidden_dim: int, output_dim: int, act: str, num_layers: int) -> nn.Sequential:
    """The pure-torch MLP body: ``num_layers - 2`` hidden blocks then a bias-free output layer."""
    layers: list[nn.Module] = []
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(make_activation(act))
    layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
    body = nn.Sequential(*layers)
    for layer in body:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
    return body


class ChartAssignmentMLP(nn.Module):
    """``x -> p(chart | x)``.

    The softmax is applied by :meth:`forward` rather than being an output activation, because two
    callers need the logits: the warmup, which trains this network against K-means labels with
    cross-entropy, and any code that wants an argmax without paying for the exponential.

    Args:
        input_dim: 3.
        output_dim: ``num_charts``.
        hidden_dim / num_layers: Body shape (``hidden_dim``, ``c_num_layers``).
        init_bias_std / init_bias: Output bias draw (``c_bias_std``, ``c_bias``).
        degree: Fourier octaves for the front end (``c_pe_degree``, 1 -- assignment should be
            smooth, so it gets far fewer than the coordinate maps).
        act / out_act: Hidden and output activations.
        use_tcnn / scale_by_pi: See :class:`FirstLayerWithZeroInit`.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 8, hidden_dim: int = 256,
                 num_layers: int = 8, init_bias_std: float = 0.0, init_bias: float = 0.0,
                 degree: int = 1, act: str = "ReLU", out_act: str = "None",
                 use_tcnn: bool = False, scale_by_pi: bool = False) -> None:
        super().__init__()
        first_layer = FirstLayerWithZeroInit(input_dim, hidden_dim, use_tcnn, degree, scale_by_pi)
        last_bias = BiasLayer(output_dim, init_bias=init_bias, init_bias_std=init_bias_std)

        if use_tcnn:
            init_dim = output_dim - 1 if output_dim > 1 else output_dim
            middle_layers = _tcnn_network(hidden_dim, output_dim, act, num_layers, init_dim)
        else:
            middle_layers = _torch_network(hidden_dim, output_dim, act, num_layers)

        self.model = nn.Sequential(first_layer, middle_layers, last_bias, make_activation(out_act))

    def forward(self, x: Tensor, return_logits: bool = False) -> Tensor:
        """``(B, input_dim) -> (B, num_charts)``, probabilities unless ``return_logits``."""
        logits = self.model(x)
        if return_logits:
            return logits
        return F.softmax(logits, dim=-1)


class MappingMLP(nn.Module):
    """One network emitting all charts' outputs, or one network per chart.

    Used twice with the axes swapped: ``texture_coordinate_mlp`` maps ``x (3) -> uv (2)`` and
    ``surface_coordinate_mlp`` maps ``uv (2) -> x (3)``. Composing them in either order is what the
    cycle losses in :mod:`uv.losses` measure.

    With ``share_mlps`` (true everywhere) the body emits ``num_charts * output_dim`` values in one
    pass and :meth:`forward` slices out the requested chart. That is a real capacity change, not
    only a speed one -- the charts share every hidden feature and differ only in the output
    projection -- and it is what the checkpoints were trained with.

    Args:
        input_dim / output_dim: 3->2 or 2->3.
        hidden_dim / num_layers: Body shape.
        init_bias_std / init_bias: Output bias draw. See :class:`BiasLayer` on why the spread
            matters for the surface map.
        degree: Fourier octaves (``t_pe_degree`` / ``s_pe_degree``, 4).
        num_charts: How many charts the shared body emits.
        act / out_act: Hidden and output activations.
        use_tcnn / scale_by_pi: See :class:`FirstLayerWithZeroInit`.
        share_mlps: One body for all charts.
        single_model: Emit ``output_dim`` values total rather than one set per chart. Only the
            (refused) texture-MLP path used it; kept so the class stays a faithful port.
    """

    def __init__(self, input_dim: int = 2, output_dim: int = 3, hidden_dim: int = 256,
                 num_layers: int = 8, init_bias_std: float = 1.0, init_bias: float = 0.0,
                 degree: int = 4, num_charts: int = 8, act: str = "ReLU", out_act: str = "None",
                 use_tcnn: bool = False, share_mlps: bool = True, single_model: bool = False,
                 scale_by_pi: bool = False) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.single_model = single_model
        self.share_mlps = share_mlps
        self.output_dim = output_dim

        if share_mlps:
            combined_output_dim = output_dim if single_model else output_dim * num_charts
            first_layer = FirstLayerWithZeroInit(input_dim, hidden_dim, use_tcnn, degree,
                                                 scale_by_pi)
            last_bias = BiasLayer(combined_output_dim, init_bias=init_bias,
                                  init_bias_std=init_bias_std)
            if use_tcnn:
                middle_layers = _tcnn_network(hidden_dim, combined_output_dim, act, num_layers,
                                              combined_output_dim - 1)
            else:
                middle_layers = _torch_network(hidden_dim, combined_output_dim, act, num_layers)
            self.model = nn.Sequential(first_layer, middle_layers, last_bias,
                                       make_activation(out_act))
        else:
            self.model = nn.ModuleList()
            for _ in range(num_charts):
                first_layer = FirstLayerWithZeroInit(input_dim, hidden_dim, use_tcnn, degree,
                                                     scale_by_pi)
                last_bias = BiasLayer(output_dim, init_bias=init_bias,
                                      init_bias_std=init_bias_std)
                if use_tcnn:
                    middle_layers = _tcnn_network(hidden_dim, output_dim, act, num_layers, None)
                else:
                    middle_layers = _torch_network(hidden_dim, output_dim, act, num_layers)
                self.model.append(nn.Sequential(first_layer, middle_layers, last_bias,
                                                make_activation(out_act)))

    def forward(self, x: Tensor, mlp_idx: int | Tensor | None = None) -> Tensor:
        """Evaluate one chart, a per-point chart, or (shared, ``mlp_idx=None``) all of them.

        Args:
            x: ``(B, input_dim)``.
            mlp_idx: ``None`` for the raw body output (shape ``(B, num_charts * output_dim)`` on the
                shared path), an ``int`` for one chart, or a ``(B,)`` index tensor to pick a
                different chart per point.

        Returns:
            ``(B, output_dim)``, except for the ``None`` case on a shared body.
        """
        if self.single_model and mlp_idx is None:
            return self.model(x)

        if self.share_mlps and mlp_idx is None:
            return self.model(x)

        if self.share_mlps:
            if isinstance(mlp_idx, int) or x.ndim == 1:
                return self.model(x).reshape(-1, self.num_charts, self.output_dim)[:, mlp_idx, :]
            output = torch.empty((x.size(0), self.output_dim), dtype=x.dtype, device=x.device)
            for idx in range(self.num_charts):
                mask = mlp_idx == idx
                if mask.sum() > 0:
                    model_output = self.model(x[mask]).type_as(output)
                    output[mask] = model_output.reshape(-1, self.num_charts,
                                                        self.output_dim)[:, idx, :]
            return output

        if isinstance(mlp_idx, int):
            return self.model[mlp_idx](x)
        output = torch.empty((x.size(0), self.output_dim), dtype=x.dtype, device=x.device)
        for idx in range(self.num_charts):
            mask = mlp_idx == idx
            if mask.sum() > 0:
                output[mask] = self.model[idx](x[mask])
        return output

    def get_all_charts_outputs(self, x: Tensor) -> list[Tensor]:
        """Every chart's output for the same input, as a list of ``(B, output_dim)``."""
        if self.share_mlps:
            all_outputs = self.model(x).reshape(-1, self.num_charts, self.output_dim)
            return [all_outputs[:, i, :] for i in range(self.num_charts)]
        return [self.model[i](x) for i in range(self.num_charts)]


class ChartOutputs(NamedTuple):
    """Everything one training step needs from the three networks, for :mod:`uv.losses`.

    ``Dti_pxs`` / ``Dti_qxs`` are absent because the conformal and stretch losses that
    consumed them are not part of this release -- both weigh 0.0 in every shipped config.
    """

    #: ``(B, num_charts)`` chart probabilities for the input points.
    chart_probs: Tensor
    #: Per chart, ``(B, 2)``: where this chart maps each input point.
    pred_uvs: list[Tensor]
    #: ``(B, 2)`` uniform samples in the unit square, the 2-3-2 cycle's starting point.
    random_uvs: Tensor
    #: Per chart, ``(B, 3)``: the input points pushed through ``uv -> x`` (the 3-2-3 cycle).
    points_3d_from_pred_uv: list[Tensor]
    #: Per chart, ``(B, 3)``: where the random UVs land on the surface.
    points_3d_from_sampled_uv: list[Tensor]
    #: Per chart, ``(B, 2)``: those points mapped back to UV (the 2-3-2 cycle).
    points_2d_from_sampled_uv: list[Tensor]
    #: Per chart, ``(B, num_charts)``: which chart the assignment network thinks those points
    #: belong to. The entropy loss asks that it be this one.
    chart_probs_points_3d_from_sampled_uv: list[Tensor]


class Nuvo(nn.Module):
    """The atlas: three chart networks, a texture map per chart, and their optimizers.

    Args:
        conf: The ``nuvo`` config node -- ``model`` / ``texture`` / ``additional_texture`` /
            ``train`` / ``loss`` / ``optimizer`` blocks.
        device: Where the directly-allocated parameters (the texture maps, the background features,
            sigma) are created. Defaults to CUDA when available.

    Raises:
        NotImplementedError: for a configuration this release does not implement -- a texture MLP,
            no texture map, or view-dependent chart assignment.
    """

    def __init__(self, conf: Any, device: str | torch.device | None = None) -> None:
        super().__init__()

        self.conf = conf
        self.num_charts = conf.model.num_charts
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _reject_unreachable(conf)

        use_tcnn = conf.model.use_tcnn
        if conf.loss.jacobian_conformal > 0.0:
            use_tcnn = False
            print("Jacobian conformal loss is used, setting use_tcnn to False")

        chart_input_dim = 3
        self.chart_assignment_mlp = ChartAssignmentMLP(
            input_dim=chart_input_dim,
            output_dim=conf.model.num_charts,
            hidden_dim=conf.model.hidden_dim,
            num_layers=conf.model.c_num_layers,
            init_bias_std=conf.model.c_bias_std,
            init_bias=conf.model.c_bias,
            degree=conf.model.c_pe_degree,
            use_tcnn=use_tcnn,
            scale_by_pi=conf.model.scale_by_pi,
        )

        self.texture_coordinate_mlp = MappingMLP(
            input_dim=3,
            output_dim=2,
            hidden_dim=conf.model.hidden_dim,
            num_layers=conf.model.num_layers,
            init_bias_std=conf.model.t_bias_std,
            init_bias=conf.model.t_bias,
            degree=conf.model.t_pe_degree,
            num_charts=conf.model.num_charts,
            use_tcnn=use_tcnn,
            out_act=conf.model.t_last_act,
            share_mlps=conf.model.t_share_mlps,
            scale_by_pi=conf.model.scale_by_pi,
        )
        print(f"Using activation function {conf.model.t_last_act} for texture coordinate MLP")

        self.surface_coordinate_mlp = MappingMLP(
            input_dim=2,
            output_dim=3,
            hidden_dim=conf.model.hidden_dim,
            num_layers=conf.model.num_layers,
            init_bias_std=conf.model.s_bias_std,
            init_bias=conf.model.s_bias,
            degree=conf.model.s_pe_degree,
            num_charts=conf.model.num_charts,
            use_tcnn=use_tcnn,
            share_mlps=conf.model.s_share_mlps,
            scale_by_pi=conf.model.scale_by_pi,
        )

        self.texture_map_res = conf.texture.texture_map_res
        self.texture_map_dim = conf.texture.texture_map_dim
        texture_map_res_per_chart = int(conf.texture.texture_map_res
                                        // (conf.model.num_charts ** 0.5))
        self.texture_map_res_per_chart = texture_map_res_per_chart
        print(f"Num charts: {conf.model.num_charts}, "
              f"Texture map resolution per chart: {texture_map_res_per_chart}")
        self.texture_map = nn.Parameter(
            _init_tensor(conf.texture.texture_map_init_func)(
                conf.model.num_charts, texture_map_res_per_chart, texture_map_res_per_chart,
                conf.texture.texture_map_dim, device=self.device),
            requires_grad=True,
        )
        self.texture_map_act = make_activation(conf.texture.texture_map_act)

        if conf.additional_texture.use:
            additional_res_per_chart = int(conf.additional_texture.texture_map_res
                                           // (conf.model.num_charts ** 0.5))
            print(f"Num charts: {conf.model.num_charts}, "
                  f"Additional texture resolution per chart: {additional_res_per_chart}")
            self.additional_texture_map = nn.Parameter(
                _init_tensor(conf.additional_texture.texture_map_init_func)(
                    conf.model.num_charts, additional_res_per_chart, additional_res_per_chart,
                    conf.additional_texture.texture_map_dim, device=self.device),
                requires_grad=True,
            )
            self.additional_texture_res = conf.additional_texture.texture_map_res
            self.additional_texture_res_per_chart = additional_res_per_chart
            self.additional_texture_dim = conf.additional_texture.texture_map_dim
            self.additional_texture_map_act = make_activation(
                conf.additional_texture.texture_map_act)
        else:
            self.additional_texture_map = None
            self.additional_texture_dim = conf.additional_texture.texture_map_dim
            self.additional_texture_map_act = None

        #: The texture MLP is refused at construction; the attribute exists because the render
        #: paths branch on it and a reader should see that the branch is dead, not absent.
        self.texture_mlp = None

        if not conf.texture.white_bkg or conf.texture.texture_map_dim > 3:
            self.texture_map_init_func = torch.zeros
        else:
            self.texture_map_init_func = torch.ones

        self.texture_map_bkg_feat = None
        if conf.texture.learnable_bkg_feat:
            print("Using learnable background texture features")
            self.texture_map_bkg_feat = nn.Parameter(
                torch.randn(self.texture_map_dim, device=self.device), requires_grad=True)

        self.additional_texture_map_init_func = torch.zeros
        self.additional_texture_map_bkg_feat = None
        if conf.additional_texture.learnable_bkg_feat:
            print("Using learnable background texture features")
            self.additional_texture_map_bkg_feat = nn.Parameter(
                torch.randn(self.additional_texture_dim, device=self.device), requires_grad=True)

        #: Target UV-cell area for the stretch loss. That loss is not ported, so this trains only
        #: through its own optimizer group and stays at 1.0 -- but it is in every archived
        #: state_dict and it keeps its group, so removing it would break checkpoint loading.
        self.sigma = nn.Parameter(torch.tensor(1.0, device=self.device))

        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
        self.init_optimizers()


    def init_optimizers(self, T_max: int | None = None) -> None:
        """Build one Adam and one cosine schedule per parameter group.

        Per-group optimizers rather than one over ``parameters()`` because the learning rates differ
        by two orders of magnitude -- ``1e-4`` for the chart networks against ``0.02`` for the
        texture map -- and because ``nuvo.train.fix_keys`` freezes a group by *removing* its
        optimizer, which only works if the groups are separate objects.

        Each schedule anneals over ``min(T_max, <group>_max_steps)``. The ``*_max_steps`` keys are
        ``1e9`` in every shipped config, so in practice ``T_max`` decides.

        Args:
            T_max: Horizon for the cosine schedules. ``None`` uses ``conf.train.iters``.
                **The launcher passes ``training.steps`` (250000), not ``nuvo.train.iters``
                (10000)**, so a released run only ever walked the first 4% of its cosine curve --
                the learning rates are effectively constant. Preserved: see :mod:`train_uv`.
        """
        self.optimizers = {}
        self.schedulers = {}

        T_max = self.conf.train.iters if T_max is None else T_max

        def add(name: str, params, lr: float) -> None:
            optimizer = torch.optim.Adam(params, lr=lr)
            max_steps = min(T_max, self.conf.train[f"{name}_max_steps"])
            print(f"{name} max steps: {max_steps}")
            self.optimizers[name] = optimizer
            self.schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_steps)

        opt = self.conf.optimizer
        add("chart_assignment", self.chart_assignment_mlp.parameters(), opt.chart_assignment_lr)
        add("texture_coordinate", self.texture_coordinate_mlp.parameters(),
            opt.texture_coordinate_lr)
        add("surface_coordinate", self.surface_coordinate_mlp.parameters(),
            opt.surface_coordinate_lr)
        add("sigma", [self.sigma], opt.sigma_lr)

        if self.texture_map is not None:
            add("texture_map", [self.texture_map], opt.texture_map_lr)
        if self.texture_mlp is not None:
            add("texture_mlp", self.texture_mlp.parameters(), opt.texture_mlp_lr)
        if self.texture_map_bkg_feat is not None:
            add("texture_map_bkg_feat", [self.texture_map_bkg_feat], opt.texture_map_bkg_feat_lr)
        if self.additional_texture_map is not None:
            add("additional_texture_map", [self.additional_texture_map],
                opt.additional_texture_map_lr)
        if self.additional_texture_map_bkg_feat is not None:
            add("additional_texture_map_bkg_feat", [self.additional_texture_map_bkg_feat],
                opt.additional_texture_map_bkg_feat_lr)

        print(f"Optimizers and schedulers initialized for {list(self.optimizers.keys())}.")

    def fix_groups(self, names) -> None:
        """Freeze parameter groups by dropping their optimizers (``nuvo.train.fix_keys``).

        A frozen group keeps its gradients -- nothing zeroes them, they are simply never applied.
        Freezing the chart assignment is the usual case: it holds an atlas partition seeded from a
        checkpoint or a warmup steady while the coordinate maps carry on training.

        Raises:
            ValueError: if a name is not one of :data:`OPTIMIZER_GROUPS`. Silently skipping unknown
                names would let a typo freeze nothing while the run looked fine, so this raises
                instead.
        """
        for name in names:
            if name not in OPTIMIZER_GROUPS:
                raise ValueError(
                    f"nuvo.train.fix_keys names {name!r}, which is not a parameter group; "
                    f"expected one of {OPTIMIZER_GROUPS}"
                )
            if name not in self.optimizers:
                print(f"Nuvo group {name} is not built by this configuration; nothing to fix")
                continue
            print("Fixing Nuvo {}".format(name))
            self.optimizers.pop(name)
            self.schedulers.pop(name)

    def zero_grad(self, set_to_none: bool = False) -> None:  # noqa: D401 - matches nn.Module
        """Zero every group's gradients, through the optimizers rather than the parameters.

        A parameter that ``fix_keys`` froze has no optimizer, so its stale gradient is left alone --
        the reason this is not ``nn.Module.zero_grad``. ``set_to_none`` is accepted for signature
        parity and ignored: each optimizer applies its own default, which on torch 2.x means
        gradients are set to ``None``.
        """
        for optimizer in self.optimizers.values():
            if optimizer is not None:
                optimizer.zero_grad()

    def step(self, scaler: Any = None, step: int = -1) -> bool:
        """Apply one update to every group that has gradients and has not run out of steps.

        A group whose parameters got no gradient this step is skipped rather than stepped with
        zeros: under AMP, stepping a gradient-free optimizer still registers an inf-check result and
        muddies the loss-scale bookkeeping.

        Args:
            scaler: The stage-A model's :class:`torch.amp.GradScaler`, shared so that one scale
                governs both models' updates. ``None`` steps unscaled.
            step: Current iteration, against the ``*_max_steps`` cut-offs. Defaults to ``-1``,
                since ``None`` would raise on the first comparison.

        Returns:
            Whether any group stepped -- what the caller checks before ``scaler.update()``.
        """
        valid_grads = True
        any_stepped = False
        for name, optimizer in self.optimizers.items():
            if optimizer is None:
                continue
            if step >= self.conf.train[f"{name}_max_steps"]:
                continue

            has_grad = any(
                param.grad is not None and param.grad.data.numel() > 0
                for param_group in optimizer.param_groups
                for param in param_group["params"]
                if param.grad is not None
            )

            if self.conf.train.skip_nan_grad:
                for param_group in optimizer.param_groups:
                    for param in param_group["params"]:
                        if param.grad is not None and torch.isnan(param.grad).any():
                            valid_grads = False
                            break

            if not valid_grads:
                print("@@@@@@@@@ Skipping optimizer step because of nan grads for {}".format(name))
                self.zero_grad()

            if has_grad and valid_grads:
                if scaler is not None:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
                any_stepped = True
            self.schedulers[name].step()
        return any_stepped


    def get_chart_probs(self, x: Tensor, rays_d: Tensor | None = None,
                        return_logits: bool = False) -> Tensor:
        """``(B, 3) -> (B, num_charts)``. ``rays_d`` is ignored (``c_use_rayd`` is refused)."""
        return self.chart_assignment_mlp(x, return_logits=return_logits)

    def get_texture_coordinates_all_charts(self, x: Tensor) -> list[Tensor]:
        """Every chart's UV for each input point."""
        return self.texture_coordinate_mlp.get_all_charts_outputs(x)

    def get_surface_coordinates_all_charts(self, x: Tensor) -> list[Tensor]:
        """Every chart's surface point for each input UV."""
        return self.surface_coordinate_mlp.get_all_charts_outputs(x)

    def get_all_charts(self, points_3d: Tensor, chart_probs: Tensor | None = None,
                       pred_uvs: list[Tensor] | None = None) -> ChartOutputs:
        """Run both cycles for every chart.

        One call produces everything the geometry losses need:

        * ``x -> uv -> x`` (the 3-2-3 cycle), per chart, for the reconstruction loss;
        * ``uv -> x -> uv`` starting from fresh uniform samples (the 2-3-2 cycle), which is what
          forces the maps to cover the whole unit square rather than collapsing to a curve;
        * the assignment network's opinion of the points the second cycle invented, which the
          entropy loss pushes toward the chart that invented them.

        The random UVs are drawn fresh every call, on the model's device, so this consumes the CUDA
        RNG stream once per training step.

        Args:
            points_3d: ``(B, 3)`` surface points. Must require grad if the Jacobian distortion loss
                is enabled -- that loss differentiates the UV map with respect to this tensor.
            chart_probs: Precomputed ``(B, num_charts)``, or ``None`` to evaluate the network.
            pred_uvs: Precomputed per-chart UVs, or ``None`` to evaluate the network.

        Returns:
            A :class:`ChartOutputs`.

        Note:
            There is no ``normals`` argument computing tangent-space UV vectors for the conformal
            and stretch losses here. Both losses weigh 0.0 in every shipped config and are not
            ported, so the argument is gone rather than silently ignored.
        """
        if chart_probs is None:
            chart_probs = self.get_chart_probs(points_3d)
        uvs = torch.rand(points_3d.shape[0], 2, device=self.device)

        if pred_uvs is None:
            pred_uvs = self.texture_coordinate_mlp.get_all_charts_outputs(points_3d)

        points_3d_from_pred_uv = [
            self.surface_coordinate_mlp(pred_uvs[i], i) for i in range(self.num_charts)
        ]
        points_3d_from_sampled_uv = self.surface_coordinate_mlp.get_all_charts_outputs(uvs)
        points_2d_from_sampled_uv = [
            self.texture_coordinate_mlp(points_3d_from_sampled_uv[i], i)
            for i in range(self.num_charts)
        ]
        chart_probs_points_3d_from_sampled_uv = [
            self.chart_assignment_mlp(points_3d_from_sampled_uv[i]) for i in range(self.num_charts)
        ]

        return ChartOutputs(
            chart_probs=chart_probs,
            pred_uvs=pred_uvs,
            random_uvs=uvs,
            points_3d_from_pred_uv=points_3d_from_pred_uv,
            points_3d_from_sampled_uv=points_3d_from_sampled_uv,
            points_2d_from_sampled_uv=points_2d_from_sampled_uv,
            chart_probs_points_3d_from_sampled_uv=chart_probs_points_3d_from_sampled_uv,
        )


    def _checkerboard(self) -> Tensor:
        """One hue-per-chart checkerboard, as a ``(num_charts, 256, 256, 3)`` tensor in [0, 255]."""
        hsv_colors = [(i / self.num_charts, 0.5, 0.5) for i in range(self.num_charts)]
        rgb_colors = [(255 * np.array(colorsys.hsv_to_rgb(*hsv))).astype(int) for hsv in hsv_colors]
        textures = checkerboard_textures((256, 256), 16, rgb_colors)
        return torch.tensor(np.stack(textures, axis=0), dtype=torch.float).to(self.device)

    def get_fused_texture_map_from_points(
        self,
        points_3d: Tensor,
        argmax: bool,
        render_checkboard: bool = False,
        pred_uvs: list[Tensor] | None = None,
        pred_chart_probs: Tensor | None = None,
        texture_map: Tensor | None = None,
        return_additional_texture_map: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Look each point's colour up in the atlas.

        Two ways to combine the charts, and they are not the same function:

        * ``argmax=False`` (training) -- every chart is sampled and the results are averaged with
          the chart probabilities. Differentiable in the probabilities, and it is what lets the
          assignment network learn, but a point near a chart boundary gets a blend of two charts'
          texels, which blurs the atlas exactly where seams are.
        * ``argmax=True`` (rendering) -- each point reads only its most likely chart. Sharp, and
          discontinuous at boundaries.

        ``texture_map_stop_uv_grad`` and ``texture_map_stop_prob_grad`` (both true in the editing
        config) detach the UVs and the probabilities here, so training the texture map cannot move
        the geometry underneath it. That is what makes stage 2 a pure image fit.

        Args:
            points_3d: ``(B, 3)``.
            argmax: Hard chart selection, as above.
            render_checkboard: Also return a per-point checkerboard colour, for visualising charts.
            pred_uvs / pred_chart_probs: Precomputed, or ``None`` to evaluate the networks.
            texture_map: Substitute atlas -- an *edited* one, already activated. ``None`` uses the
                model's own, passed through ``texture_map_act``.
            return_additional_texture_map: Also sample the second atlas at the same UVs.

        Returns:
            ``(fused, checkerboard | None, additional | None)``; ``fused`` is ``(B, C)``.
        """
        chart_probs = self.get_chart_probs(points_3d) if pred_chart_probs is None \
            else pred_chart_probs
        if self.conf.texture.texture_map_stop_prob_grad:
            chart_probs = chart_probs.detach()

        acted_texture_map = self.texture_map_act(self.texture_map) if texture_map is None \
            else texture_map
        num_points = points_3d.shape[0]
        fused_texture_map = torch.zeros(num_points, acted_texture_map.shape[-1],
                                        device=self.device)
        fused_additional_texture_map = None
        if self.additional_texture_map is not None and return_additional_texture_map:
            fused_additional_texture_map = torch.zeros(num_points, self.additional_texture_dim,
                                                       device=self.device)

        fused_checkboard_color = None
        checkboard_textures = None
        if render_checkboard:
            checkboard_textures = self._checkerboard()
            fused_checkboard_color = torch.zeros(num_points, 3, device=self.device)

        interp = self.conf.texture.interpolation_mode
        padding = self.conf.texture.padding_mode

        if argmax:
            chart_indices = torch.argmax(chart_probs, dim=1)
            uvs = self.texture_coordinate_mlp(points_3d, chart_indices)
            if self.conf.texture.texture_map_stop_uv_grad:
                uvs = uvs.detach()

            for chart_idx in range(self.num_charts):
                mask = chart_indices == chart_idx
                if mask.sum() == 0:
                    continue
                chart_uvs = uvs[mask]
                if self.conf.texture.texture_map_stop_uv_grad:
                    chart_uvs = chart_uvs.detach()
                fused_texture_map[mask] = bilinear_interpolation(
                    acted_texture_map[chart_idx], chart_uvs, interp, padding)

                if fused_additional_texture_map is not None:
                    chart_additional = self.additional_texture_map_act(
                        self.additional_texture_map[chart_idx])
                    fused_additional_texture_map[mask] = bilinear_interpolation(
                        chart_additional, chart_uvs,
                        self.conf.additional_texture.interpolation_mode,
                        self.conf.additional_texture.padding_mode)

                if render_checkboard:
                    fused_checkboard_color[mask] = bilinear_interpolation(
                        checkboard_textures[chart_idx], chart_uvs, interp, padding)
        else:
            all_chart_uvs = self.texture_coordinate_mlp.get_all_charts_outputs(points_3d) \
                if pred_uvs is None else pred_uvs

            for chart_idx in range(self.num_charts):
                chart_uvs = all_chart_uvs[chart_idx]
                if self.conf.texture.texture_map_stop_uv_grad:
                    chart_uvs = chart_uvs.detach()
                weight = chart_probs[:, chart_idx].unsqueeze(-1)
                fused_texture_map = fused_texture_map + weight * bilinear_interpolation(
                    acted_texture_map[chart_idx], chart_uvs, interp, padding)

                if fused_additional_texture_map is not None:
                    chart_additional = self.additional_texture_map_act(
                        self.additional_texture_map[chart_idx])
                    fused_additional_texture_map = fused_additional_texture_map + \
                        weight * bilinear_interpolation(
                            chart_additional, chart_uvs,
                            self.conf.additional_texture.interpolation_mode,
                            self.conf.additional_texture.padding_mode)

                if render_checkboard:
                    fused_checkboard_color = fused_checkboard_color + \
                        weight * bilinear_interpolation(
                            checkboard_textures[chart_idx], chart_uvs, interp, padding)

        return fused_texture_map, fused_checkboard_color, fused_additional_texture_map

    def get_rgb_from_texture_map_for_points(
        self,
        points_3d: Tensor,
        rays_o: Tensor | None = None,
        rays_d: Tensor | None = None,
        normals: Tensor | None = None,
        argmax: bool = True,
        render_checkboard: bool = False,
        pred_uvs: list[Tensor] | None = None,
        pred_chart_probs: Tensor | None = None,
        texture_map: Tensor | None = None,
        return_additional_texture_map: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Colour for a flat batch of surface points.

        With the texture-MLP path refused this is :meth:`get_fused_texture_map_from_points` with the
        ``rays_o`` / ``rays_d`` / ``normals`` arguments kept, though only the view-dependent
        (unreachable) paths ever read them. Stage 2 passes zero tensors for them.
        """
        return self.get_fused_texture_map_from_points(
            points_3d, argmax, render_checkboard, pred_uvs=pred_uvs,
            pred_chart_probs=pred_chart_probs, texture_map=texture_map,
            return_additional_texture_map=return_additional_texture_map,
        )

    def get_rgb_for_texture_map(self, grid_len: int | None = None) -> Tensor:
        """The atlas as an image stack.

        This is what gets exported for editing and re-imported afterwards, so the resampling
        matters: ``align_corners=True`` matches :func:`bilinear_interpolation`, which means a
        texel's UV does not move when the atlas is resized and an edit made at one resolution lands
        in the same place at another.

        Args:
            grid_len: Resample each chart to this side length. ``None``, or the atlas's own
                resolution, returns it untouched.

        Returns:
            ``(num_charts, H, W, C)`` after ``texture_map_act``.
        """
        texture_map = self.texture_map_act(self.texture_map)
        if grid_len is not None and self.texture_map_res_per_chart != grid_len:
            texture_map = F.interpolate(
                texture_map.permute(0, 3, 1, 2), size=(grid_len, grid_len),
                mode=self.conf.texture.interpolation_mode, align_corners=True,
            ).permute(0, 2, 3, 1)
        return texture_map

    def get_rgb_for_additional_texture_map(self, grid_len: int | None = None) -> Tensor | None:
        """The second atlas as an image stack, or ``None`` when it is not built."""
        if self.additional_texture_map is None:
            return None
        additional = self.additional_texture_map_act(self.additional_texture_map)
        if grid_len is not None and self.additional_texture_res_per_chart != grid_len:
            additional = F.interpolate(
                additional.permute(0, 3, 1, 2), size=(grid_len, grid_len),
                mode=self.conf.additional_texture.interpolation_mode, align_corners=True,
            ).permute(0, 2, 3, 1)
        return additional

    def load_model(self, state_dict, exclude_keys=()) -> list[str]:
        """Copy what fits from a checkpoint, reporting what did not.

        Tolerant on purpose: a Nuvo checkpoint is routinely loaded into a model with a different
        atlas resolution or chart count, and the useful failure mode is "the chart networks loaded,
        the texture map did not" rather than an exception.

        Args:
            state_dict: Parameter mapping from a checkpoint.
            exclude_keys: Substrings; any parameter whose name contains one is skipped -- how to
                seed the chart assignment from one checkpoint while leaving the rest fresh.

        Returns:
            The names that were skipped, in checkpoint order.
        """
        own_state = self.state_dict()
        skipped: list[str] = []
        for name, param in state_dict.items():
            if any(key in name for key in exclude_keys):
                print(f"Nuvo parameter {name} skipped: excluded by key")
                skipped.append(name)
            elif name not in own_state:
                print(f"Parameter {name} not found in the state_dict")
                skipped.append(name)
            elif own_state[name].shape != param.shape:
                print(f"Nuvo parameter {name} skipped: shape mismatch "
                      f"(current: {own_state[name].shape}, loaded: {param.shape})")
                skipped.append(name)
            else:
                own_state[name].copy_(param)
        return skipped


def _init_tensor(name: str):
    """The allocator a ``texture_map_init_func`` names. Anything unrecognised means ``randn``."""
    if name == "zeros":
        return torch.zeros
    if name == "ones":
        return torch.ones
    return torch.randn


def _reject_unreachable(conf: Any) -> None:
    """Refuse the configurations this release does not implement, at construction.

    Every one of these is off in both shipped UV configs. Refusing beats silently ignoring: a
    config that asks for a texture MLP and gets a texture map back would train, converge, and be
    wrong.
    """
    if conf.texture.use_texture_mlp:
        raise NotImplementedError(
            "nuvo.texture.use_texture_mlp is not implemented in this release; the reachable path "
            "is the per-chart texture map (use_texture_map: true, use_texture_mlp: false)"
        )
    if not conf.texture.use_texture_map:
        raise NotImplementedError(
            "nuvo.texture.use_texture_map: false requires the texture MLP, which is not "
            "implemented in this release"
        )
    if conf.model.c_use_rayd:
        raise NotImplementedError(
            "nuvo.model.c_use_rayd (view-dependent chart assignment) is not implemented in this "
            "release; the chart assignment is a function of position only"
        )
    for name in ("conformal", "stretch"):
        if conf.loss[name] > 0:
            raise NotImplementedError(
                f"nuvo.loss.{name} > 0 needs the tangent-space UV vectors (surface normals and "
                f"per-point Jacobians), which are not ported; use nuvo.loss.jacobian_conformal "
                f"instead -- it is the normal-free distortion term the paper uses"
            )

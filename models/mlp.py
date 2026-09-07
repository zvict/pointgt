from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from models.activations import make_activation


def _validate_indices(name: str, values: Sequence[int], *, upper: int) -> tuple[int, ...]:
    """Reject layer indices that name no layer, rather than ignoring them at build time."""
    out = tuple(int(v) for v in values)
    if len(set(out)) != len(out):
        raise ValueError(f"{name} contains duplicate indices: {out}")
    for value in out:
        if not 0 <= value <= upper:
            raise ValueError(f"{name} index {value} is out of range for 0..{upper}")
    return out


class MLP(nn.Module):
    """Fully connected network with optional input skips, external residuals, and halved stages.

    Args:
        inp_dim: Width of the input tensor's last dimension.
        num_layers: Number of *hidden* layers; the module builds ``num_layers + 1`` linear layers.
        num_channels: Hidden width.
        out_dim: Width of the output tensor's last dimension.
        act_type: Activation after every layer but the last. See
            :func:`models.activations.make_activation`.
        last_act_type: Activation after the final layer.
        bias: Whether the linear layers carry a bias.
        skip_layers: Layer indices whose input is concatenated with the network input.
        half_layers: Boundary indices at which the running width is halved. Boundary ``k`` halves
            both the output of layer ``k - 1`` and the input of layer ``k``, so it may equal
            ``num_layers + 1`` to halve the output.
        residual_layers: Layer indices whose input is concatenated with a tensor supplied to
            :meth:`forward`.
        residual_dims: Width of each of those tensors, one per entry in ``residual_layers``.
        use_weight_norm: Unsupported; see below.
        inplace: Forwarded to the activation factory.

    Raises:
        ValueError: for a malformed layer index or a non-positive layer width.
        NotImplementedError: if ``use_weight_norm`` is set. Torch ships two weight-norm
            implementations whose ``state_dict`` keys differ (``weight_g``/``weight_v`` versus
            ``parametrizations.weight.original0``/``original1``), so the choice silently renames
            parameters. No shipped config enables it and no archived checkpoint contains a
            weight-normalised layer, so rather than ship an untested branch that could rename keys,
            the flag is rejected.
    """

    def __init__(
        self,
        inp_dim: int,
        num_layers: int,
        num_channels: int,
        out_dim: int,
        *,
        act_type: str | None = "relu",
        last_act_type: str | None = "none",
        bias: bool = True,
        skip_layers: Sequence[int] = (),
        half_layers: Sequence[int] = (),
        residual_layers: Sequence[int] = (),
        residual_dims: Sequence[int] = (),
        use_weight_norm: bool = False,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        if use_weight_norm:
            raise NotImplementedError(
                "use_weight_norm is not supported: torch's two weight-norm implementations use "
                "different state_dict key names, and no released config or checkpoint uses it"
            )
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")
        for label, value in (("inp_dim", inp_dim), ("num_channels", num_channels),
                             ("out_dim", out_dim)):
            if value <= 0:
                raise ValueError(f"{label} must be positive, got {value}")

        num_linear = num_layers + 1
        self._skip_layers = _validate_indices("skip_layers", skip_layers, upper=num_linear - 1)
        self._residual_layers = _validate_indices(
            "residual_layers", residual_layers, upper=num_linear - 1
        )
        self._half_layers = _validate_indices("half_layers", half_layers, upper=num_linear)
        self._residual_dims = tuple(int(d) for d in residual_dims)
        if len(self._residual_dims) != len(self._residual_layers):
            raise ValueError(
                f"residual_dims has {len(self._residual_dims)} entries but residual_layers has "
                f"{len(self._residual_layers)}"
            )

        self.inp_dim = int(inp_dim)
        self.num_layers = int(num_layers)
        self.num_channels = int(num_channels)
        self.out_dim = int(out_dim)

        half = set(self._half_layers)
        layers: list[nn.Module] = [nn.Identity()]
        for i in range(num_linear):
            layer_in = inp_dim if i == 0 else num_channels
            layer_out = out_dim if i == num_linear - 1 else num_channels
            if i + 1 in half:
                layer_out //= 2
            if i in half:
                layer_in //= 2
            if i in self._skip_layers:
                layer_in += inp_dim
            if i in self._residual_layers:
                layer_in += self._residual_dims[self._residual_layers.index(i)]
            if layer_in <= 0 or layer_out <= 0:
                raise ValueError(
                    f"layer {i} would be Linear({layer_in}, {layer_out}); check half_layers"
                )
            layers.append(nn.Linear(layer_in, layer_out, bias=bias))
            is_last = i == num_linear - 1
            layers.append(
                make_activation(last_act_type if is_last else act_type, inplace=inplace)
            )
        self.model = nn.ModuleList(layers)

        for parameter in self.model.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

        self._skip_positions = frozenset(2 * i + 1 for i in self._skip_layers)
        self._residual_positions = {2 * i + 1: n for n, i in enumerate(self._residual_layers)}

    @property
    def linear_layers(self) -> tuple[nn.Linear, ...]:
        """The linear layers in depth order. Recomputed, so it registers no duplicate modules."""
        return tuple(m for m in self.model if isinstance(m, nn.Linear))

    @property
    def num_linear_layers(self) -> int:
        return self.num_layers + 1

    def forward(self, x: Tensor, residuals: Sequence[Tensor] = ()) -> Tensor:
        """Apply the network.

        Args:
            x: ``(..., inp_dim)`` input.
            residuals: One tensor per entry in ``residual_layers``, in that order, each
                ``(..., residual_dims[i])`` and broadcast-compatible with the activation at that
                layer.

        Returns:
            ``(..., out_dim)``.
        """
        if len(residuals) != len(self._residual_layers):
            raise ValueError(
                f"expected {len(self._residual_layers)} residual tensors, got {len(residuals)}"
            )
        if x.shape[-1] != self.inp_dim:
            raise ValueError(f"expected last dimension {self.inp_dim}, got {tuple(x.shape)}")

        network_input = x
        for position, layer in enumerate(self.model):
            if position in self._skip_positions:
                x = torch.cat([x, network_input], dim=-1)
            slot = self._residual_positions.get(position)
            if slot is not None:
                x = torch.cat([x, residuals[slot]], dim=-1)
            x = layer(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"inp_dim={self.inp_dim}, num_layers={self.num_layers}, "
            f"num_channels={self.num_channels}, out_dim={self.out_dim}"
        )

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from models.papr import PAPR

__all__ = ["get_model"]


def get_model(args: Any, device: torch.device | str) -> PAPR:
    """Build the PAPR model described by ``args`` on ``device``.

    ``args`` is the resolved config (``utils.config.load_config(...).values``), not a namespace of
    command-line flags -- the model reads ``args.geoms``, ``args.models`` and ``args.training``
    directly, so the whole tree has to be handed over rather than a flattened subset.

    ``device`` is passed to the constructor *and* the built model is moved onto it, so the caller
    gets a model that is ready to render and never has to place it itself.

    Both halves are needed. ``PAPR`` takes ``device`` because it allocates several tensors there
    directly, but the submodules built through plain ``torch.nn`` constructors land on the CPU
    regardless: on ``configs/nerfsyn/lego.yml``, 35 of the 51 ``state_dict`` entries -- the whole
    U-Net, the k and q attention networks, the point cloud itself -- are still on the CPU when the
    constructor returns, and the first render would raise a device mismatch. The ``.to()`` is what
    makes the signature true. It copies values unchanged and touches neither RNG stream, so it is
    numerics-neutral wherever it is called from.

    Sibling factory :func:`uv.get_nuvo` is built the same way, for the same reason.
    """
    from models.papr import PAPR

    return PAPR(args, device).to(device)

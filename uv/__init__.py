from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from uv.nuvo import Nuvo

__all__ = ["get_nuvo"]


def get_nuvo(args: Any, device: torch.device | str = "cuda") -> Nuvo:
    """Build the UV atlas described by ``args`` on ``device``.

    Args:
        args: The ``nuvo`` config node -- ``utils.config.load_config(...).values.nuvo`` -- not the
            whole tree and not a namespace of flags. The atlas reads ``args.model``,
            ``args.texture``, ``args.additional_texture``, ``args.train``, ``args.loss`` and
            ``args.optimizer``.
        device: Where the directly-allocated parameters (texture maps, background features, sigma)
            are created. The chart networks are built on the CPU by ``torch.nn`` and moved here, so
            the returned module is fully on ``device``.

    Returns:
        A :class:`uv.nuvo.Nuvo`, already moved to ``device``, with its optimizers built for
        ``nuvo.train.iters``. ``train_uv.py`` rebuilds them with the stage-A horizon; see there.
    """
    from uv.nuvo import Nuvo

    return Nuvo(args, device=device).to(device)

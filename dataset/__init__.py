from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch.utils.data import DataLoader

    from dataset.dataset import RINDataset
    from utils.runtime import ConfigNode

__all__ = ["get_dataset", "get_loader"]


def get_dataset(
    args: ConfigNode, mode: str, device: str | torch.device = "cuda"
) -> RINDataset:
    """Build the dataset for one split.

    Args:
        args: The ``dataset`` config node -- ``args.dataset`` for training, ``eval.dataset`` merged
            over it for validation, or one entry of ``test.datasets`` for testing.
        mode: ``"train"`` or ``"test"``; picks the transforms file, and gates GT depth loading.
        device: Where the split lives. The whole split is resident, so this is the training device.
    """
    from dataset.dataset import RINDataset

    if mode not in ("train", "test"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'train' or 'test'")
    return RINDataset(args, mode=mode, device=device)


def get_loader(dataset: RINDataset, args: ConfigNode, mode: str) -> DataLoader:
    """Wrap a dataset in the loader its split uses.

    Training shuffles into batches of ``args.batch_size``; testing walks the split in order, one
    frame at a time.

    ``num_workers`` is never passed, so loading always happens in the main process -- ``args``
    carries the key but it is unused. That is not an oversight: the dataset's tensors already
    live on the GPU, and a forked worker cannot touch them.
    """
    import torch
    from torch.utils.data import DataLoader

    if mode == "train":
        generator = torch.Generator()
        generator.manual_seed(torch.initial_seed())
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            generator=generator,
        )
    if mode == "test":
        return DataLoader(dataset, batch_size=1, shuffle=False)
    raise ValueError(f"Unknown mode {mode!r}; expected 'train' or 'test'")

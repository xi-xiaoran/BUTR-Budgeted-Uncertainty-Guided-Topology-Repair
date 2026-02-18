from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from data import build_dataset
from data.base import DSItem

def collate_dsitems(batch):
    """Collate a list[DSItem] into a single DSItem with batched tensors.

    This is required because PyTorch's default_collate cannot handle arbitrary
    dataclass instances (DSItem). Keep this as a top-level function so it is
    picklable on Windows when num_workers > 0.
    """
    images = torch.stack([b.image for b in batch], dim=0)
    masks  = torch.stack([b.mask  for b in batch], dim=0)
    metas  = [b.meta for b in batch]
    return DSItem(images, masks, metas)

def make_loaders(dataset_name, data_root, crop, batch_size, num_workers=2):
    tr = build_dataset(dataset_name, data_root, "train", crop=crop)
    te = build_dataset(dataset_name, data_root, "test",  crop=None)  # eval on full images
    dl_tr = DataLoader(
        tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_dsitems,
    )
    dl_te = DataLoader(
        te,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dsitems,
    )
    return dl_tr, dl_te

def to_device(batch, device):
    # batch is DSItem; meta is a list[dict] after collation
    return batch.image.to(device), batch.mask.to(device), batch.meta

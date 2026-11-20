"""CIFAR loaders sourced from Hugging Face `datasets` with a local numpy cache."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .paths import DATA_DIR

CIFAR_MEAN = (0.5071, 0.4867, 0.4402)
CIFAR_STD = (0.2675, 0.2565, 0.2761)
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


class ArrayDataset(Dataset):
    """Wraps uint8 HWC arrays so augmentation happens per sample."""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])


def _hf_arrays(name, split, root):
    cache = Path(root) / f"{name}_{split}.npy"
    labels_cache = Path(root) / f"{name}_{split}_labels.npy"
    if cache.exists() and labels_cache.exists():
        return np.load(cache, mmap_mode="r"), np.load(labels_cache)

    from datasets import load_dataset

    ds = load_dataset(f"uoft-cs/{name}", split=split)
    label_key = "fine_label" if name == "cifar100" else "label"
    images = np.stack([np.asarray(r["img"], dtype=np.uint8) for r in ds])
    labels = np.asarray(ds[label_key], dtype=np.int64)
    Path(root).mkdir(parents=True, exist_ok=True)
    np.save(cache, images)
    np.save(labels_cache, labels)
    return images, labels


def make_transforms(resolution=32, train=True):
    ops = []
    if resolution != 32:
        ops.append(transforms.Resize((resolution, resolution)))
    if train:
        ops += [
            transforms.RandomCrop(resolution, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    return transforms.Compose(ops)


def make_loaders(name, batch_size=128, num_workers=4, max_train=None, max_val=None,
                 resolution=32, seed=42, root=None, drop_last=False):
    """Returns (train_loader, val_loader) with deterministic subsets."""
    root = root or DATA_DIR
    train_imgs, train_lbls = _hf_arrays(name, "train", root)
    val_imgs, val_lbls = _hf_arrays(name, "test", root)

    rng = np.random.default_rng(seed)
    tr_idx = rng.permutation(len(train_lbls))
    va_idx = rng.permutation(len(val_lbls))
    if max_train:
        tr_idx = tr_idx[:max_train]
    if max_val:
        va_idx = va_idx[:max_val]

    train_ds = ArrayDataset(train_imgs[tr_idx], train_lbls[tr_idx], make_transforms(resolution, True))
    val_ds = ArrayDataset(val_imgs[va_idx], val_lbls[va_idx], make_transforms(resolution, False))

    pin = torch.cuda.is_available()
    common = dict(num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=drop_last, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader

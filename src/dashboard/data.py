"""Dataset: scikit-learn's bundled digits set as PyTorch tensors.

We use ``load_digits`` (1,797 8x8 images, ships with scikit-learn) so the whole
project runs in seconds on CPU with no download — the focus here is *training
dynamics and tooling*, not squeezing out state-of-the-art accuracy.
"""

from __future__ import annotations

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42


def get_dataloaders(batch_size: int = 64, seed: int = RANDOM_STATE):
    """Return ``(train_loader, val_loader, meta)``.

    Pixels are scaled to [0, 1]; features and labels become float/long tensors.
    """
    digits = load_digits()
    X = digits.data.astype("float32") / 16.0   # digits pixels are 0..16
    y = digits.target.astype("int64")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    meta = {"n_features": X.shape[1], "n_classes": int(y.max() + 1)}
    return train_loader, val_loader, meta

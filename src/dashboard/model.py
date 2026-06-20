"""A small configurable MLP — the thing whose training dynamics we study."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with a configurable hidden width and depth."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: int = 128,
        depth: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)  # reproducible initialisation across runs
        layers: list[nn.Module] = []
        prev = n_features
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

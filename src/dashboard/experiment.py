"""A single tracked training run.

`run_experiment` trains one model under one `ExperimentConfig` and logs rich
diagnostics to TensorBoard each epoch:

  * train/val loss and accuracy        -> learning progress and overfitting
  * learning rate                       -> verify the schedule
  * gradient norm                       -> spot exploding/vanishing gradients
  * weight histograms                   -> watch the parameter distribution

It also returns a plain `history` dict so we can make static comparison plots and
write tests without parsing TensorBoard event files.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .data import get_dataloaders
from .model import MLP


@dataclass
class ExperimentConfig:
    """Everything that defines one run. Vary one field to run an ablation."""
    name: str = "run"
    optimizer: str = "adam"          # "adam" | "sgd" | "sgd_momentum"
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 30
    hidden: int = 128
    depth: int = 2
    seed: int = 0


def _make_optimizer(cfg: ExperimentConfig, params) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr)
    if cfg.optimizer == "sgd_momentum":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


def _grad_norm(model: nn.Module) -> float:
    """Global L2 norm of all gradients — a key training-health signal."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm() ** 2)
    return total ** 0.5


def run_experiment(
    cfg: ExperimentConfig,
    log_root: str | Path = "runs",
    write_tb: bool = True,
) -> dict[str, list[float]]:
    """Train one configuration; log to TensorBoard; return its history."""
    train_loader, val_loader, meta = get_dataloaders(cfg.batch_size, seed=cfg.seed)
    model = MLP(meta["n_features"], meta["n_classes"],
                hidden=cfg.hidden, depth=cfg.depth, seed=cfg.seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(cfg, model.parameters())

    writer = SummaryWriter(Path(log_root) / cfg.name) if write_tb else None
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "grad_norm": [],
    }

    for epoch in range(cfg.epochs):
        # ---- train ----
        model.train()
        tl = tc = n = 0.0
        last_grad = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            last_grad = _grad_norm(model)   # measured before the step
            optimizer.step()
            tl += loss.item() * xb.size(0)
            tc += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
        train_loss, train_acc = tl / n, tc / n

        # ---- validate ----
        model.eval()
        vl = vc = vn = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                vl += criterion(logits, yb).item() * xb.size(0)
                vc += (logits.argmax(1) == yb).sum().item()
                vn += xb.size(0)
        val_loss, val_acc = vl / vn, vc / vn

        # NaN guard: a diverging run produces NaN losses; record and stop early.
        if not torch.isfinite(torch.tensor(train_loss)):
            history["train_loss"].append(float("nan"))
            break

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["grad_norm"].append(last_grad)

        if writer is not None:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("grad_norm", last_grad, epoch)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)

    if writer is not None:
        writer.add_hparams(
            {k: v for k, v in asdict(cfg).items() if k != "name"},
            {"hparam/best_val_acc": max(history["val_acc"], default=0.0)},
        )
        writer.close()
    return history

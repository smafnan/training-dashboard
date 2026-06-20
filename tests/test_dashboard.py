"""Tests for the experiment-tracking and ablation tooling."""

from __future__ import annotations

import math

import torch

from dashboard import ExperimentConfig, MLP, get_dataloaders, run_experiment


def test_dataloaders_shapes():
    train, val, meta = get_dataloaders(batch_size=32)
    xb, yb = next(iter(train))
    assert xb.shape[1] == meta["n_features"] == 64
    assert meta["n_classes"] == 10
    assert xb.dtype == torch.float32 and yb.dtype == torch.int64


def test_model_forward_shape():
    model = MLP(64, 10, hidden=32, depth=2)
    out = model(torch.randn(8, 64))
    assert out.shape == (8, 10)


def test_good_config_learns(tmp_path):
    cfg = ExperimentConfig(name="good", optimizer="adam", lr=1e-3, epochs=15)
    h = run_experiment(cfg, log_root=tmp_path, write_tb=False)
    # Loss should drop substantially and accuracy should be high.
    assert h["train_loss"][-1] < h["train_loss"][0] * 0.5
    assert h["val_acc"][-1] > 0.85


def test_history_records_all_signals(tmp_path):
    cfg = ExperimentConfig(name="signals", epochs=3)
    h = run_experiment(cfg, log_root=tmp_path, write_tb=False)
    for key in ["train_loss", "train_acc", "val_loss", "val_acc", "grad_norm"]:
        assert len(h[key]) == 3
    assert all(g >= 0 for g in h["grad_norm"])


def test_too_high_lr_diverges(tmp_path):
    # A wildly high LR should NOT produce a good model; loss stays high or NaNs.
    bad = ExperimentConfig(name="bad", optimizer="sgd_momentum", lr=5.0, epochs=15)
    good = ExperimentConfig(name="good", optimizer="sgd_momentum", lr=3e-2, epochs=15)
    hb = run_experiment(bad, log_root=tmp_path, write_tb=False)
    hg = run_experiment(good, log_root=tmp_path, write_tb=False)
    best_bad = max(hg["val_acc"]) if False else max(hb["val_acc"], default=0.0)
    # The diverging run is clearly worse than the well-tuned one.
    assert best_bad < max(hg["val_acc"]) - 0.2


def test_tensorboard_event_files_written(tmp_path):
    cfg = ExperimentConfig(name="tb", epochs=2)
    run_experiment(cfg, log_root=tmp_path, write_tb=True)
    events = list((tmp_path / "tb").glob("events.out.tfevents.*"))
    assert len(events) >= 1

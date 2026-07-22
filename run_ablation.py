"""Run a one-variable ablation and chart the effect.

By default this varies the **learning rate** across five values — including one
that is far too high — so the loss curves visibly show the three failure/​success
modes you must be able to diagnose:

  * too low  -> loss falls but crawls; underfits in the epoch budget
  * good     -> loss falls fast and smoothly to a low value
  * too high -> loss oscillates / plateaus high / diverges to NaN

    python run_ablation.py                 # learning-rate ablation
    python run_ablation.py --ablation optimizer

Everything is logged to ./runs/ for TensorBoard, and static comparison figures +
a results table are written to ./reports/ so the finding is visible on GitHub.

    tensorboard --logdir runs              # then open http://localhost:6006
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dashboard import ExperimentConfig, run_experiment

# The ablation grids: each maps a run name -> the field override(s).
ABLATIONS = {
    "lr": [
        ("lr_3e-3", {"optimizer": "sgd_momentum", "lr": 3e-3}),
        ("lr_3e-2", {"optimizer": "sgd_momentum", "lr": 3e-2}),
        ("lr_1e-1", {"optimizer": "sgd_momentum", "lr": 1e-1}),
        ("lr_5e-1", {"optimizer": "sgd_momentum", "lr": 5e-1}),
        ("lr_3e0",  {"optimizer": "sgd_momentum", "lr": 3.0}),   # far too high
    ],
    "optimizer": [
        ("sgd",          {"optimizer": "sgd", "lr": 1e-1}),
        ("sgd_momentum", {"optimizer": "sgd_momentum", "lr": 1e-1}),
        ("adam",         {"optimizer": "adam", "lr": 1e-3}),
    ],
    "batch_size": [
        ("bs_16",  {"batch_size": 16}),
        ("bs_64",  {"batch_size": 64}),
        ("bs_256", {"batch_size": 256}),
    ],
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", choices=list(ABLATIONS), default="lr")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--log-root", default="runs")
    p.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name, overrides in ABLATIONS[args.ablation]:
        cfg = ExperimentConfig(name=f"{args.ablation}/{name}",
                               epochs=args.epochs, **overrides)
        print(f"running {cfg.name} ...")
        history = run_experiment(cfg, log_root=args.log_root)
        best = max(history["val_acc"], default=float("nan"))
        results[name] = {"history": history, "best_val_acc": best,
                         "overrides": overrides}
        print(f"   best val acc = {best:.4f}  "
              f"(final train loss = {history['train_loss'][-1]:.4f})")

    _plot_comparison(results, args.ablation,
                     args.output_dir / f"ablation_{args.ablation}.png")
    summary = {k: {"best_val_acc": v["best_val_acc"],
                   "overrides": v["overrides"]} for k, v in results.items()}
    (args.output_dir / f"ablation_{args.ablation}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary (best validation accuracy):")
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]["best_val_acc"]):
        print(f"   {k:14s} {v['best_val_acc']:.4f}")
    print(f"\nFigures in {args.output_dir}/  |  TensorBoard: tensorboard --logdir {args.log_root}")
    return 0


def _plot_comparison(results: dict, ablation: str, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, r in results.items():
        h = r["history"]
        ax1.plot(h["train_loss"], label=name)
        ax2.plot(h["val_acc"], label=name)
    ax1.set_title(f"Training loss vs epoch ({ablation} ablation)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss"); ax1.legend(fontsize=8)
    ax1.set_yscale("log")
    ax2.set_title("Validation accuracy vs epoch")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("val accuracy"); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

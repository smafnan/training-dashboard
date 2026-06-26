# Training-Dynamics Dashboard — Experiment Tracking + a Learning-Rate Ablation

> **AI Engineer Roadmap — Project 2.3**
> *Teaches: rigorous experimentation, reproducibility, reading loss curves.*
> *Done when: you can look at a loss curve and diagnose the problem.*

A small PyTorch MLP wrapped in proper **experiment tracking** (TensorBoard) plus a
controlled **ablation**: change exactly one thing — the learning rate — and chart
the effect. The result is a side-by-side set of loss curves that show every
failure mode you need to recognise on sight.

```bash
python -m venv .venv && source .venv/bin/activate   # Win: .\.venv\Scripts\activate
pip install -e ".[dev]"

python run_ablation.py --ablation lr        # the headline experiment
python run_ablation.py --ablation optimizer # bonus: SGD vs momentum vs Adam
python run_ablation.py --ablation batch_size

tensorboard --logdir runs                   # interactive dashboard @ localhost:6006
pytest -q                                   # 6 tests
```

Everything runs in seconds on CPU (scikit-learn digits, no download). Each run
logs to `runs/` for TensorBoard, and static comparison figures land in `reports/`.

---

## 🖥️ Web UI (React + Tailwind + FastAPI)

An interactive dashboard ships with the project: pick the optimizer, learning
rate, and epochs, hit **Train**, and watch the **loss curve** and **validation
accuracy** draw live (the MLP trains in a couple of seconds). Hit **Run ablation**
to overlay several learning rates and see a too-high one flatline near 10% — the
signature of divergence.

```bash
pip install -e ".[web]"
uvicorn api:app --reload          # open http://localhost:8000

# (optional) rebuild / develop the frontend:
cd web && npm install && npm run build
```

The committed `web/dist` means `uvicorn api:app` works straight from a clone.

## The ablation: learning rate (the "Done when")

Five SGD+momentum runs, identical except for the learning rate, 40 epochs each:

![Learning-rate ablation](reports/ablation_lr.png)

| Learning rate | Best val accuracy | What the loss curve looks like — and the diagnosis |
| --- | ---: | --- |
| `3e-3` (too low) | 0.933 | Loss falls but **crawls**; still descending at epoch 40 → **underfitting the budget**. Fix: raise the LR or train longer. |
| `3e-2` | 0.975 | Smooth, healthy descent. |
| **`1e-1` (just right)** | **0.978** | **Fast, smooth** drop to train loss ≈ 0.0007 → well-tuned. |
| `5e-1` (too high) | 0.733 | Loss **oscillates and plateaus high** (~2.0) → steps overshoot the minimum. Fix: lower the LR. |
| `3e0` (far too high) | 0.103 ≈ random | Loss **diverges / stays at chance** (10 classes → 0.10) → training is broken. Fix: lower the LR by 10–100×. |

**This is the whole skill:** given only the left-hand loss-curve plot, you can
name the problem (too low / good / too high / diverged) and prescribe the fix
without seeing the accuracy numbers. The too-high run flatlining at the
random-guess accuracy of 0.10 is the unmistakable signature of divergence.

---

## What gets tracked

Every run logs these to TensorBoard each epoch — the signals you actually use to
debug training:

| Signal | What it tells you |
| --- | --- |
| `loss/train` vs `loss/val` | learning progress; a widening gap = overfitting |
| `accuracy/train` vs `accuracy/val` | the same, in accuracy terms |
| `lr` | confirms the schedule is doing what you think |
| `grad_norm` | exploding (→∞) or vanishing (→0) gradients |
| weight histograms | parameter distributions drifting or saturating |

Open `tensorboard --logdir runs` and you can scrub all of these interactively,
overlaying every run in the ablation.

## Reproducibility

- A single `ExperimentConfig` dataclass fully specifies a run; an ablation is
  just "the same config with one field swept".
- Seeds are fixed for data split, shuffling, and model init, so runs are
  comparable and repeatable.
- `add_hparams` logs each config + its best val accuracy to TensorBoard's HParams
  tab for at-a-glance comparison.

## Layout

```
src/dashboard/
├── data.py         # sklearn digits as torch DataLoaders (no download)
├── model.py        # small configurable MLP
└── experiment.py   # ExperimentConfig + run_experiment (TensorBoard logging)
run_ablation.py     # sweep one hyperparameter; write curves + JSON summary
tests/              # 6 tests incl. "good config learns" and "too-high LR diverges"
```

## License

MIT.

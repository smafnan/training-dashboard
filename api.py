"""FastAPI backend for the Training Dashboard web UI.

Runs a small MLP training on demand (sklearn digits, a few seconds) and returns
the per-epoch history so the frontend can chart it live. Also exposes a quick
learning-rate ablation. Serves the built React frontend.

Run:  uvicorn api:app --reload  →  http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.dashboard import ExperimentConfig, run_experiment

ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Training Dashboard API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class TrainRequest(BaseModel):
    optimizer: str = "sgd_momentum"   # adam | sgd | sgd_momentum
    lr: float = 0.1
    epochs: int = 30
    batch_size: int = 64


def _history(cfg: ExperimentConfig) -> dict:
    h = run_experiment(cfg, write_tb=False)
    return {
        "loss": [round(x, 4) for x in h["train_loss"]],
        "val_acc": [round(x, 4) for x in h["val_acc"]],
        "final_val_acc": round(h["val_acc"][-1], 4) if h["val_acc"] else 0.0,
        "min_loss": round(min(h["train_loss"]), 4) if h["train_loss"] else 0.0,
        "epochs": len(h["train_loss"]),
    }


@app.post("/api/train")
def train(req: TrainRequest):
    cfg = ExperimentConfig(
        name="ui", optimizer=req.optimizer, lr=req.lr,
        epochs=max(1, min(req.epochs, 60)), batch_size=req.batch_size,
    )
    return _history(cfg)


@app.get("/api/ablation")
def ablation():
    """A quick learning-rate sweep: too-low / good / too-high / diverging."""
    grid = [("lr 3e-3", 3e-3), ("lr 1e-1", 1e-1), ("lr 5e-1", 5e-1), ("lr 3.0", 3.0)]
    runs = {}
    for label, lr in grid:
        cfg = ExperimentConfig(name=label, optimizer="sgd_momentum", lr=lr, epochs=30)
        runs[label] = _history(cfg)
    return {"runs": runs}


_dist = ROOT / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")

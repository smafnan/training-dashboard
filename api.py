"""FastAPI backend for the Training Dashboard web UI.

Runs a small MLP training on demand (sklearn digits, a few seconds) and returns
the per-epoch history so the frontend can chart it live. Also exposes a quick
learning-rate ablation. Serves the built React frontend.

Run:  uvicorn api:app --reload  →  http://localhost:8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dashboard import ExperimentConfig, run_experiment

ROOT = Path(__file__).resolve().parent

# Allowlist of origins the frontend is served from; override with a comma-separated
# ALLOWED_ORIGINS env var for non-local deployments. Local dev works with no env set.
_DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:8000"
_allowed_origins = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]

app = FastAPI(title="Training Dashboard API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins, allow_methods=["*"],
                   allow_headers=["*"])


class TrainRequest(BaseModel):
    optimizer: Literal["adam", "sgd", "sgd_momentum"] = "sgd_momentum"
    lr: float = 0.1
    epochs: int = 30
    batch_size: int = Field(default=64, gt=0, le=1024)


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
    try:
        return _history(cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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

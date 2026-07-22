"""Tests for the FastAPI backend (api.py): /api/train input validation."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_train_happy_path_returns_history():
    resp = client.post("/api/train", json={"optimizer": "adam", "epochs": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["epochs"] == 2
    assert len(body["loss"]) == 2


def test_train_rejects_unknown_optimizer():
    resp = client.post("/api/train", json={"optimizer": "not_a_real_optimizer"})
    assert resp.status_code == 422


def test_train_rejects_out_of_range_batch_size():
    resp = client.post("/api/train", json={"batch_size": 0})
    assert resp.status_code == 422

    resp = client.post("/api/train", json={"batch_size": 10_000})
    assert resp.status_code == 422

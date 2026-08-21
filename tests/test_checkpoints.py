"""Tests for atomic metadata-backed checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.checkpoints import (
    atomic_write_bytes,
    load_best_score,
    meta_path_for,
    save_best_score,
    write_checkpoint_metadata,
)


def test_atomic_write_replaces_existing(tmp_path: Path):
    target = tmp_path / "latest_model.zip"
    atomic_write_bytes(target, b"v1")
    assert target.read_bytes() == b"v1"
    atomic_write_bytes(target, b"v2-longer")
    assert target.read_bytes() == b"v2-longer"


def test_interrupted_temp_does_not_clobber(tmp_path: Path, monkeypatch):
    target = tmp_path / "latest_model.zip"
    atomic_write_bytes(target, b"good")

    def _boom(*_args, **_kwargs):
        raise OSError("simulated interrupt")

    monkeypatch.setattr("training.checkpoints.atomic_replace", _boom)
    with pytest.raises(OSError, match="simulated interrupt"):
        atomic_write_bytes(target, b"bad")
    assert target.read_bytes() == b"good"
    assert list(tmp_path.glob(".latest_model.zip.*.tmp")) == []


def test_best_score_roundtrip(tmp_path: Path):
    assert load_best_score(tmp_path) is None
    save_best_score(tmp_path, -1.5)
    assert load_best_score(tmp_path) == pytest.approx(-1.5)


def test_best_score_persists_metric_name(tmp_path: Path):
    from training.checkpoints import load_best_score_record

    save_best_score(tmp_path, 42.5, metric="mean_makespan")
    record = load_best_score_record(tmp_path)
    assert record is not None
    assert record["best_metric"] == "mean_makespan"
    assert record["best_score"] == pytest.approx(42.5)
    assert "best_mean_makespan" not in record
    assert load_best_score(tmp_path) == pytest.approx(42.5)


def test_checkpoint_metadata_writes_config_and_zip_hash(tmp_path: Path):
    ckpt = tmp_path / "latest_model.zip"
    ckpt.write_bytes(b"zip")
    cfg = {"env": {"n_machines": 5}, "seed": 1}
    write_checkpoint_metadata(ckpt, config=cfg)
    meta = json.loads(meta_path_for(ckpt).read_text(encoding="utf-8"))
    assert meta["config"] == cfg
    assert "config_fingerprint" not in meta
    assert "zip_sha256" in meta


def test_zip_hash_is_bound_to_bytes_at_save(tmp_path: Path):
    from training.checkpoints import file_sha256

    ckpt = tmp_path / "latest_model.zip"
    ckpt.write_bytes(b"zip-a")
    cfg = {"env": {"n_machines": 5}}
    write_checkpoint_metadata(ckpt, config=cfg)
    meta = json.loads(meta_path_for(ckpt).read_text(encoding="utf-8"))
    assert meta["zip_sha256"] == file_sha256(ckpt)
    ckpt.write_bytes(b"zip-b")
    assert meta["zip_sha256"] != file_sha256(ckpt)

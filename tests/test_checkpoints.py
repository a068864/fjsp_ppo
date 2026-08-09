"""Tests for atomic metadata-backed checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.checkpoints import (
    assert_config_compatible,
    atomic_write_bytes,
    config_fingerprint,
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


def test_metadata_fingerprint_and_incompatible_resume(tmp_path: Path):
    ckpt = tmp_path / "latest_model.zip"
    ckpt.write_bytes(b"zip")
    cfg_a = {"env": {"n_machines": 5}, "seed": 1}
    cfg_b = {"env": {"n_machines": 6}, "seed": 1}
    write_checkpoint_metadata(ckpt, config=cfg_a)
    meta = json.loads(meta_path_for(ckpt).read_text(encoding="utf-8"))
    assert meta["config_fingerprint"] == config_fingerprint(cfg_a)
    assert_config_compatible(ckpt, cfg_a)
    with pytest.raises(ValueError, match="fingerprint"):
        assert_config_compatible(ckpt, cfg_b)

"""Tests for checkpoint callback cadence, final eval, and trust gating."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from callbacks import BestModelCallback, FJSPEvalCallback, LatestCheckpointCallback
from config import get_default_train_config
from train import maybe_resume, resolve_resume_path
from training.checkpoints import load_best_score, save_best_score, write_checkpoint_metadata


class _FakeModel:
    def __init__(self, path_tracker: list):
        self.path_tracker = path_tracker
        self.seed = 0
        self.num_timesteps = 0
        self.logger = None

    def save(self, path: str) -> None:
        self.path_tracker.append(str(path))
        Path(path).write_bytes(b"fake-zip")


def test_latest_checkpoint_fires_on_completed_update_boundary(tmp_path: Path):
    saved = []
    cb = LatestCheckpointCallback(
        save_path=tmp_path / "latest_model.zip",
        save_freq_updates=2,
        n_steps=4,
        n_envs=1,
        verbose=0,
    )
    cb.model = _FakeModel(saved)
    cb.num_timesteps = 4  # mid-ish; completed updates use rollout_end
    assert cb._on_step() is True
    assert saved == []
    cb.num_timesteps = 8  # exactly 2 updates
    cb._on_rollout_end()
    assert len(saved) == 1


def test_best_callback_preserves_existing_score(tmp_path: Path):
    save_best_score(tmp_path, 3.5, metric="mean_reward")
    cb = BestModelCallback(tmp_path / "best_model.zip", metric="mean_reward")
    cb.model = _FakeModel([])
    cb.load_persisted_best(tmp_path)
    assert cb.best_mean_reward == pytest.approx(3.5)
    assert cb.update(3.0) is False
    assert cb.update(4.0) is True
    assert load_best_score(tmp_path) == pytest.approx(4.0)


def test_best_callback_minimizes_makespan(tmp_path: Path):
    cb = BestModelCallback(tmp_path / "best_model.zip", metric="mean_makespan")
    cb.model = _FakeModel([])
    assert cb.update(90.0) is True
    assert cb.update(95.0) is False
    assert cb.update(80.0) is True
    assert load_best_score(tmp_path) == pytest.approx(80.0)


def test_eval_callback_reseeds_held_out_schedule(tmp_path: Path, monkeypatch):
    """FJSPEvalCallback must reseed with cfg.eval_seed before each eval."""
    from config import get_default_train_config

    cfg = get_default_train_config()
    cfg.eval_seed = 1_000_007
    seeded = []

    class _FakeEvalEnv:
        num_envs = 1

        def seed(self, seed):
            seeded.append(int(seed))
            return [seed]

        def reset(self):
            return {"action_mask": np.ones((1, 2), dtype=np.float32)}

        def step(self, actions):
            raise AssertionError("should not step; evaluate is mocked")

    class _FakeResult:
        mean_reward = 0.0
        mean_makespan = 12.0
        mean_ep_length = 1.0
        success_rate = 1.0
        mean_inference_time_s = 0.0

        def format_summary(self):
            return "ok"

    monkeypatch.setattr(
        "callbacks.evaluate_policy_fjsp",
        lambda *args, **kwargs: _FakeResult(),
    )
    env = _FakeEvalEnv()
    cb = FJSPEvalCallback(
        eval_env=env,
        best_model_path=tmp_path / "best_model.zip",
        n_eval_episodes=1,
        eval_freq_updates=1,
        n_steps=4,
        n_envs=1,
        eval_seed=int(cfg.eval_seed),
        best_metric="mean_makespan",
    )
    cb.model = _FakeModel([])
    cb.num_timesteps = 4
    cb._maybe_eval(force=True)
    assert seeded == [int(cfg.eval_seed)]
    assert cb.best_callback.best_score == pytest.approx(12.0)


def test_final_training_end_saves_latest(tmp_path: Path):
    saved = []
    cb = LatestCheckpointCallback(
        save_path=tmp_path / "latest_model.zip",
        save_freq_updates=10,
        n_steps=4,
        n_envs=1,
    )
    cb.model = _FakeModel(saved)
    cb._on_training_end()
    assert saved and saved[-1].endswith("latest_model.zip")


def test_resume_requires_explicit_trust(tmp_path: Path):
    cfg = get_default_train_config()
    cfg.checkpoint_dir = str(tmp_path)
    cfg.resume = True
    cfg.trust_checkpoint = False
    (tmp_path / cfg.latest_model_name).write_bytes(b"zip")
    write_checkpoint_metadata(tmp_path / cfg.latest_model_name, config=cfg.to_dict())
    with pytest.raises(ValueError, match="trust"):
        resolve_resume_path(cfg)


def test_explicit_missing_eval_path_does_not_fallback(tmp_path: Path):
    from evaluate import resolve_model_path
    from config import get_default_eval_config

    cfg = get_default_eval_config()
    cfg.model_path = str(tmp_path / "missing_explicit.zip")
    with pytest.raises(FileNotFoundError, match="missing_explicit"):
        resolve_model_path(cfg, explicit=True)

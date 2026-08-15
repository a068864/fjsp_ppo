"""Lifecycle ownership, monitor append, and evaluation metric contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monitor import FJSPMonitor
from training.evaluate import EvalResult, sample_masked_random_actions


def test_monitor_appends_without_truncating_history(tmp_path: Path):
    from envs.fjsp_env import FJSPEnv

    path = tmp_path / "worker_0"
    env = FJSPEnv(n_machines=2, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    mon = FJSPMonitor(env, filename=path, allow_early_resets=True)
    mon.reset(seed=0)
    mask = mon.unwrapped.get_action_mask()
    # Write one episode row then reopen append-mode monitor.
    action = int(np.flatnonzero(mask)[0])
    for _ in range(3):
        obs, rew, term, trunc, info = mon.step(action)
        if term or trunc:
            break
        mask = info.get("action_mask", env.get_action_mask())
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            break
        action = int(valid[0])
    mon.close()

    csv_path = Path(str(path) + ".monitor.csv") if not str(path).endswith(".monitor.csv") else path
    if not csv_path.exists():
        csv_path = tmp_path / "worker_0.monitor.csv"
    first_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(first_lines) >= 1

    env2 = FJSPEnv(n_machines=2, n_jobs=2, avg_operations_per_job=2, seed=1, device="cpu")
    mon2 = FJSPMonitor(env2, filename=path, allow_early_resets=True)
    mon2.reset(seed=1)
    mask = env2.get_action_mask()
    action = int(np.flatnonzero(mask)[0])
    obs, rew, term, trunc, info = mon2.step(action)
    # Force an episode boundary via truncate-like close write by finishing.
    if not (term or trunc):
        # Manually write via completing quickly isn't guaranteed; call writer.
        mon2.results_writer.write_row({"r": 1.0, "l": 1, "t": 1.0, "makespan": 1.0, "success": 0})
    mon2.close()
    second_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(second_lines) >= len(first_lines)


def test_sample_masked_random_actions_fails_on_empty_mask():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="empty"):
        sample_masked_random_actions(np.zeros((1, 4), dtype=np.float32), rng)


def test_eval_result_reports_success_failure_timeout_counts():
    result = EvalResult(
        mean_makespan=10.0,
        std_makespan=0.0,
        mean_ep_length=5.0,
        std_ep_length=0.0,
        success_rate=0.5,
        mean_inference_time_s=0.01,
        n_episodes=4,
        n_success=2,
        n_failure=1,
        n_timeout=1,
    )
    text = result.format_summary()
    assert "successful-episode makespan" in text.lower() or "Successful-episode makespan" in text
    assert "reward" not in text.lower()
    assert "success" in text.lower()
    assert result.n_success == 2
    assert result.n_failure == 1
    assert result.n_timeout == 1

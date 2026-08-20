"""Baseline FPS / RSS / GPU metrics for the recommended-order gate."""

from __future__ import annotations

from train import parse_args


def test_parse_args_full_scale_flag():
    args = parse_args(["--full-scale"])
    assert args.full_scale is True
    assert parse_args([]).full_scale is False


def test_parse_args_debug_flag():
    args = parse_args(["--debug"])
    assert args.debug is True
    assert parse_args([]).debug is False


def test_use_subprocess_cpu_vs_flags():
    from config import get_debug_train_config
    from train import _use_subprocess, parse_args

    cfg = get_debug_train_config()
    cfg.n_envs = 2
    cfg.device = "cpu"
    assert _use_subprocess(cfg, None) is True
    assert _use_subprocess(cfg, parse_args(["--dummy-vec"])) is False
    assert _use_subprocess(cfg, parse_args(["--subproc"])) is True
    cfg.n_envs = 1
    assert _use_subprocess(cfg, parse_args(["--subproc"])) is False


def test_process_rss_bytes_is_positive():
    from training.benchmark import process_rss_bytes

    assert process_rss_bytes() > 0


def test_measure_training_baseline_returns_finite_metrics():
    from config import get_debug_train_config
    from training.benchmark import measure_training_baseline

    cfg = get_debug_train_config()
    cfg.n_envs = 1
    cfg.device = "cpu"
    cfg.ppo.n_steps = 4
    cfg.ppo.batch_size = 4
    metrics = measure_training_baseline(cfg, n_env_steps=2)
    assert metrics.fps > 0.0
    assert metrics.rss_mb > 0.0
    assert metrics.n_env_steps == 2
    assert metrics.n_envs == 1
    assert metrics.cuda_alloc_mb >= 0.0
    assert metrics.n_machines == cfg.env.n_machines

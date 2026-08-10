"""Shared CLI helpers for evaluate.py and baseline_random.py."""

from __future__ import annotations

import argparse

from config import EvalConfig, TrainConfig, get_default_train_config


def add_shared_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add episode/seed/env-size flags shared by eval entrypoints."""
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-machines", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--avg-ops",
        type=int,
        default=None,
        dest="avg_operations_per_job",
    )


def apply_shared_eval_args(cfg: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    """Apply shared CLI overrides onto an EvalConfig and revalidate."""
    if getattr(args, "n_episodes", None) is not None:
        cfg.n_episodes = int(args.n_episodes)
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "n_machines", None) is not None:
        cfg.env.n_machines = int(args.n_machines)
    if getattr(args, "n_jobs", None) is not None:
        cfg.env.n_jobs = int(args.n_jobs)
    if getattr(args, "avg_operations_per_job", None) is not None:
        cfg.env.avg_operations_per_job = int(args.avg_operations_per_job)
    cfg.validate()
    return cfg


def build_eval_train_config(eval_cfg: EvalConfig) -> TrainConfig:
    """Build a TrainConfig carrying eval env settings for ``make_vec_env``."""
    train_cfg = get_default_train_config()
    train_cfg.seed = eval_cfg.seed
    # Offline eval uses the CLI/eval seed as the held-out episode base.
    train_cfg.eval_seed = eval_cfg.seed
    train_cfg.device = eval_cfg.device
    train_cfg.env = eval_cfg.env
    train_cfg.model = eval_cfg.model
    train_cfg.n_envs = 1
    train_cfg.normalize_reward = False
    return train_cfg


__all__ = [
    "add_shared_eval_args",
    "apply_shared_eval_args",
    "build_eval_train_config",
]

"""Shared CLI helpers for evaluate.py and baseline entrypoints."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Callable, Optional

from config import (
    EvalConfig,
    TrainConfig,
    get_debug_train_config,
    get_default_eval_config,
    get_full_scale_eval_config,
)
from utils import configure_root_logging, set_global_seed


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
    parser.add_argument(
        "--full-scale",
        action="store_true",
        help="Use FULL_SCALE_ENV (25x15x8) and matching model dims",
    )


def eval_config_from_args(args: Optional[argparse.Namespace] = None) -> EvalConfig:
    """Pick demo-scale or full-scale eval defaults from CLI flags."""
    if args is not None and getattr(args, "full_scale", False):
        return get_full_scale_eval_config()
    return get_default_eval_config()


def apply_shared_eval_args(cfg: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    """Apply shared CLI overrides onto an EvalConfig and revalidate."""
    if getattr(args, "full_scale", False):
        full = get_full_scale_eval_config()
        cfg.env = replace(full.env)
        cfg.model = replace(full.model)
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
    train_cfg = get_debug_train_config()
    train_cfg.seed = eval_cfg.seed
    # Offline eval uses the CLI/eval seed as the held-out episode base.
    train_cfg.eval_seed = eval_cfg.seed
    train_cfg.device = eval_cfg.device
    train_cfg.env = eval_cfg.env
    train_cfg.model = eval_cfg.model
    train_cfg.n_envs = 1
    return train_cfg


def prepare_eval_config(
    cfg: Optional[EvalConfig] = None,
    args: Optional[argparse.Namespace] = None,
    *,
    apply_fn: Callable[[EvalConfig, argparse.Namespace], EvalConfig] = apply_shared_eval_args,
) -> EvalConfig:
    """Load defaults, apply CLI overrides, and seed RNGs for an eval entrypoint."""
    configure_root_logging()
    cfg = cfg or eval_config_from_args(args)
    if args is not None:
        cfg = apply_fn(cfg, args)
    set_global_seed(cfg.seed, deterministic=True)
    return cfg


def make_eval_vec_env(eval_cfg: EvalConfig):
    """Build the in-process eval VecEnv used by evaluate.py and baselines."""
    from training.make_env import make_vec_env

    return make_vec_env(
        build_eval_train_config(eval_cfg),
        n_envs=1,
        use_subprocess=False,
        for_eval=True,
    )


__all__ = [
    "add_shared_eval_args",
    "apply_shared_eval_args",
    "build_eval_train_config",
    "eval_config_from_args",
    "make_eval_vec_env",
    "prepare_eval_config",
]

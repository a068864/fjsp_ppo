"""Measure FJSP PPO env-step FPS, RSS, and GPU memory/util.

Usage:
    python benchmark.py
    python benchmark.py --full-scale --n-env-steps 64
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from config import get_default_train_config, get_full_scale_train_config
from train import apply_args, parse_args
from training.benchmark import measure_training_baseline
from utils import configure_root_logging


def parse_benchmark_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse benchmark CLI; reuses train flags plus ``--n-env-steps``."""
    parser = argparse.ArgumentParser(description="FJSP PPO rollout baseline")
    parser.add_argument("--n-env-steps", type=int, default=64)
    # Import train's flags by parsing a compatible argv through train.parse_args
    # after we strip our extra flag.
    args, rest = parser.parse_known_args(argv)
    train_args = parse_args(rest)
    train_args.n_env_steps = int(args.n_env_steps)
    return train_args


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    configure_root_logging()
    args = parse_benchmark_args(argv)
    cfg = (
        get_full_scale_train_config()
        if getattr(args, "full_scale", False)
        else get_default_train_config()
    )
    cfg = apply_args(cfg, args)
    metrics = measure_training_baseline(cfg, n_env_steps=int(args.n_env_steps))
    print("=" * 60)
    print("FJSP PPO rollout baseline")
    print("=" * 60)
    print(metrics.format_summary())
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

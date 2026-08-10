"""Evaluate FJSP with an exact makespan MILP (PuLP + CBC).

Usage:
    python baseline_milp.py
    python baseline_milp.py --n-episodes 5 --seed 42
    python baseline_milp.py --time-limit 30
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from config import EvalConfig, get_default_eval_config
from training.eval_cli import (
    add_shared_eval_args,
    apply_shared_eval_args,
    build_eval_train_config,
)
from training.evaluate import evaluate_milp_fjsp, print_eval_result
from training.make_env import make_vec_env
from utils import configure_root_logging, set_global_seed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse MILP baseline CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate FJSP with an exact makespan MILP (PuLP+CBC)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="CBC wall-clock limit in seconds per instance (default: none)",
    )
    add_shared_eval_args(parser)
    return parser.parse_args(argv)


def run_baseline(
    cfg: Optional[EvalConfig] = None,
    args: Optional[argparse.Namespace] = None,
):
    """Run MILP evaluation and print metrics."""
    configure_root_logging()
    cfg = cfg or get_default_eval_config()
    time_limit = None
    if args is not None:
        cfg = apply_shared_eval_args(cfg, args)
        time_limit = getattr(args, "time_limit", None)

    set_global_seed(cfg.seed, deterministic=True)

    train_cfg = build_eval_train_config(cfg)
    eval_env = make_vec_env(
        train_cfg,
        n_envs=1,
        use_subprocess=False,
        for_eval=True,
    )

    try:
        result = evaluate_milp_fjsp(
            eval_env,
            n_episodes=cfg.n_episodes,
            seed=cfg.seed,
            time_limit=time_limit,
        )
        print_eval_result(result, title="FJSP MILP Baseline Results")
        return result
    finally:
        eval_env.close()


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    run_baseline(get_default_eval_config(), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

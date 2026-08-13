"""Evaluate FJSP with uniform random valid actions (no checkpoint).

Usage:
    python baseline_random.py
    python baseline_random.py --n-episodes 20 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from config import EvalConfig, get_default_eval_config, get_full_scale_eval_config
from training.eval_cli import (
    add_shared_eval_args,
    apply_shared_eval_args,
    build_eval_train_config,
)
from training.evaluate import evaluate_random_fjsp, print_eval_result
from training.make_env import make_vec_env
from utils import configure_root_logging, set_global_seed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse random baseline CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate FJSP with uniform random valid actions",
    )
    add_shared_eval_args(parser)
    return parser.parse_args(argv)


def run_baseline(
    cfg: Optional[EvalConfig] = None,
    args: Optional[argparse.Namespace] = None,
):
    """Run random-action evaluation and print metrics."""
    configure_root_logging()
    cfg = cfg or get_default_eval_config()
    if args is not None:
        cfg = apply_shared_eval_args(cfg, args)

    set_global_seed(cfg.seed, deterministic=True)

    train_cfg = build_eval_train_config(cfg)
    eval_env = make_vec_env(
        train_cfg,
        n_envs=1,
        use_subprocess=False,
        for_eval=True,
    )

    try:
        eval_env.seed(cfg.seed)
        result = evaluate_random_fjsp(
            eval_env,
            n_episodes=cfg.n_episodes,
            seed=cfg.seed,
        )
        print_eval_result(result, title="FJSP Random Baseline Results")
        return result
    finally:
        eval_env.close()


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    cfg = (
        get_full_scale_eval_config()
        if getattr(args, "full_scale", False)
        else get_default_eval_config()
    )
    run_baseline(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

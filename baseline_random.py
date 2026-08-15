"""Evaluate FJSP with uniform random valid actions (no checkpoint).

Usage:
    python baseline_random.py
    python baseline_random.py --n-episodes 20 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from config import EvalConfig
from training.eval_cli import (
    add_shared_eval_args,
    eval_config_from_args,
    make_eval_vec_env,
    prepare_eval_config,
)
from training.evaluate import evaluate_random_fjsp, print_eval_result


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
    cfg = prepare_eval_config(cfg, args)
    eval_env = make_eval_vec_env(cfg)

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
    run_baseline(eval_config_from_args(args), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

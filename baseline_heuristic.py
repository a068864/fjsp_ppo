"""Evaluate FJSP with classic dispatching-rule heuristics (no checkpoint).

Usage:
    python baseline_heuristic.py --rule SPT
    python baseline_heuristic.py --rule MWKR --n-episodes 20 --seed 0
    python baseline_heuristic.py --all --n-episodes 5
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Optional

from config import EvalConfig
from heuristics import RULES
from training.eval_cli import (
    add_shared_eval_args,
    eval_config_from_args,
    make_eval_vec_env,
    prepare_eval_config,
)
from training.evaluate import EvalResult, evaluate_heuristic_fjsp, print_eval_result


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse heuristic baseline CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate FJSP with classic dispatching-rule heuristics",
    )
    parser.add_argument(
        "--rule",
        type=str,
        default=None,
        help=f"Dispatching rule ({', '.join(sorted(RULES))})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every rule and print a comparison table",
    )
    add_shared_eval_args(parser)
    args = parser.parse_args(argv)
    if not args.all and args.rule is None:
        parser.error("one of --rule or --all is required")
    if args.rule is not None:
        key = args.rule.upper()
        if key not in RULES:
            parser.error(f"unknown rule {args.rule!r}; choose from {sorted(RULES)}")
        args.rule = key
    return args


def _print_comparison(results: Dict[str, EvalResult]) -> None:
    """Print a compact per-rule comparison table."""
    print("=" * 72)
    print(f"{'Rule':<8} {'Reward':>12} {'Makespan':>12} {'Success':>10} {'EpLen':>10}")
    print("-" * 72)
    for rule, result in results.items():
        mk = result.mean_makespan
        mk_s = f"{mk:.2f}" if mk != float("inf") else "inf"
        print(
            f"{rule:<8} {result.mean_reward:12.4f} {mk_s:>12} "
            f"{result.success_rate:10.2%} {result.mean_ep_length:10.1f}"
        )
    print("=" * 72)


def run_baseline(
    cfg: Optional[EvalConfig] = None,
    args: Optional[argparse.Namespace] = None,
):
    """Run heuristic evaluation and print metrics."""
    cfg = prepare_eval_config(cfg, args)
    eval_env = make_eval_vec_env(cfg)

    try:
        if args is not None and args.all:
            results: Dict[str, EvalResult] = {}
            for rule in sorted(RULES):
                # Same instance stream start per rule (VecEnv seed applies on reset).
                eval_env.seed(cfg.seed)
                result = evaluate_heuristic_fjsp(
                    eval_env,
                    rule=rule,
                    n_episodes=cfg.n_episodes,
                )
                results[rule] = result
                print_eval_result(result, title=f"FJSP Heuristic Baseline ({rule})")
            _print_comparison(results)
            return results

        rule = args.rule if args is not None else "SPT"
        eval_env.seed(cfg.seed)
        result = evaluate_heuristic_fjsp(
            eval_env,
            rule=rule,
            n_episodes=cfg.n_episodes,
        )
        print_eval_result(result, title=f"FJSP Heuristic Baseline ({rule})")
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

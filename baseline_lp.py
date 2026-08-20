"""Evaluate FJSP with LP relaxation + rounding/list scheduling.

Usage:
    python baseline_lp.py
    python baseline_lp.py --n-episodes 5 --seed 42 --rounding-trials 20
    python baseline_lp.py --compare-milp --n-episodes 5
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional, Sequence

from config import EvalConfig
from training.eval_cli import (
    add_shared_eval_args,
    eval_config_from_args,
    make_eval_vec_env,
    prepare_eval_config,
)
from training.evaluate import (
    LpEpisodeRecord,
    evaluate_lp_rounding_fjsp,
    print_eval_result,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse LP-rounding baseline CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FJSP with an LP relaxation + rounding/list-scheduling "
            "baseline"
        ),
    )
    parser.add_argument(
        "--rounding-trials",
        type=int,
        default=20,
        help="Largest-fraction plus randomized rounding trials (default: 20)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="CBC LP wall-clock limit in seconds per instance (default: none)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print LP diagnostics (status, violations, fractional counts)",
    )
    parser.add_argument(
        "--compare-milp",
        action="store_true",
        help="Also solve the exact MILP once per instance for OPT / gap columns",
    )
    add_shared_eval_args(parser)
    return parser.parse_args(argv)


def _fmt(value: Optional[float], *, width: int = 10, prec: int = 3) -> str:
    if value is None or not _finite(value):
        return f"{'n/a':>{width}}"
    return f"{value:{width}.{prec}f}"


def _finite(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def print_lp_episode_table(
    records: Sequence[LpEpisodeRecord],
    *,
    compare_milp: bool = False,
    verbose: bool = False,
) -> None:
    """Print per-instance LP bound and feasible Cmax."""
    headers = [
        "ep",
        "seed",
        "LB_LP",
        "Cmax",
        "sec",
        "trials",
        "rule",
    ]
    if compare_milp:
        headers.extend(["OPT", "LP-gap", "sol-gap"])
    if verbose:
        headers.extend(["frac", "viol", "assign"])
    print("-" * 72)
    print("Per-instance LP relaxation + rounded schedule")
    print(" ".join(f"{h:>10}" for h in headers))
    for rec in records:
        cells: List[str] = [
            f"{rec.episode:10d}",
            f"{rec.seed:10d}",
            _fmt(rec.lp_lower_bound),
            _fmt(rec.makespan),
            _fmt(rec.runtime_s, prec=4),
            f"{rec.rounding_trials:10d}",
            f"{rec.best_rule:>10}",
        ]
        if compare_milp:
            cells.extend(
                [
                    _fmt(rec.opt),
                    _fmt(rec.lp_to_opt_gap),
                    _fmt(rec.sol_to_opt_gap),
                ]
            )
        if verbose:
            cells.extend(
                [
                    f"{rec.n_fractional:10d}",
                    _fmt(rec.max_constraint_violation, prec=2),
                    _fmt(rec.assignment_sum_violation, prec=2),
                ]
            )
        print(" ".join(cells))
        if verbose:
            print(
                f"    lp_status={rec.lp_status} "
                f"n_fractional={rec.n_fractional} "
                f"max_violation={rec.max_constraint_violation:.3e} "
                f"assign_sum_violation={rec.assignment_sum_violation:.3e}"
            )
    print("-" * 72)


def run_baseline(
    cfg: Optional[EvalConfig] = None,
    args: Optional[argparse.Namespace] = None,
):
    """Run LP-rounding evaluation and print metrics."""
    cfg = prepare_eval_config(cfg, args)
    rounding_trials = getattr(args, "rounding_trials", 20) if args is not None else 20
    time_limit = getattr(args, "time_limit", None) if args is not None else None
    verbose = bool(getattr(args, "verbose", False)) if args is not None else False
    compare_milp = bool(getattr(args, "compare_milp", False)) if args is not None else False
    eval_env = make_eval_vec_env(cfg)

    try:
        result, records = evaluate_lp_rounding_fjsp(
            eval_env,
            n_episodes=cfg.n_episodes,
            seed=cfg.seed,
            rounding_trials=int(rounding_trials),
            time_limit=time_limit,
            compare_milp=compare_milp,
        )
        print_eval_result(result, title="FJSP LP-Rounding Baseline Results")
        print_lp_episode_table(
            records, compare_milp=compare_milp, verbose=verbose
        )
        return result, records
    finally:
        eval_env.close()


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    run_baseline(eval_config_from_args(args), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

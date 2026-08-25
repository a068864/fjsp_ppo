"""Compare a PPO checkpoint against random, heuristic, MILP, and LP baselines.

Usage:
    python compare_baselines.py --trust-checkpoint
    python compare_baselines.py --trust-checkpoint --seed 1000042 --n-episodes 20
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, List, Optional, Tuple

from config import EvalConfig
from evaluate import resolve_model_path
from heuristics import RULES
from models.graph_ppo import load_graph_ppo
from solvers.list_scheduler import LIST_RULES
from training.eval_cli import (
    add_shared_eval_args,
    apply_shared_eval_args,
    eval_config_from_args,
    make_eval_vec_env,
    prepare_eval_config,
)
from training.evaluate import (
    EvalResult,
    evaluate_heuristic_fjsp,
    evaluate_lp_rounding_fjsp,
    evaluate_milp_fjsp,
    evaluate_policy_fjsp,
    evaluate_random_fjsp,
)
from utils import get_device, get_logger, set_global_seed

logger = get_logger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse comparison CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare PPO against random, heuristic, MILP, and LP-rounding baselines",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model zip (default: ./checkpoints/best_model.zip)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="auto | cpu | mps | cuda | cuda:0",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Required to load an SB3/cloudpickle checkpoint ZIP",
    )
    parser.add_argument(
        "--rounding-trials",
        type=int,
        default=20,
        help="LP-rounding trials (default: 20)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="CBC wall-clock limit in seconds per instance for MILP and LP",
    )
    parser.add_argument(
        "--skip-milp",
        action="store_true",
        help="Skip exact MILP (slow on full-scale); still runs PPO/random/heuristics/LP",
    )
    add_shared_eval_args(parser)
    return parser.parse_args(argv)


def apply_compare_args(cfg: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    """Apply shared eval flags plus checkpoint path / device."""
    cfg = apply_shared_eval_args(cfg, args)
    if getattr(args, "model_path", None) is not None:
        cfg.model_path = str(args.model_path)
    if getattr(args, "device", None) is not None:
        cfg.device = str(args.device)
    cfg.validate()
    return cfg


def _fmt_mk(value: float) -> str:
    if not math.isfinite(value):
        return "inf"
    return f"{value:.3f}"


def _episode_time_s(row: Dict[str, object]) -> float:
    """Mean wall time per instance, in seconds.

    Constructive methods (PPO / random / heuristics) store per-step latency in
    ``mean_inference_time_s``. MILP and LP-rounding store whole-episode solver
    time in the same field. Multiply the former by episode length so the table
    compares one quantity.
    """
    latency = float(row["mean_inference_time_s"])
    if row.get("time_is_per_episode"):
        return latency
    return latency * float(row["mean_ep_length"])


def _print_table(rows: List[Dict[str, object]]) -> None:
    print("=" * 78)
    print("FJSP PPO vs baselines (same seed/instance stream, classic Cmax)")
    print("=" * 78)
    print(
        f"{'Method':<16} {'Makespan':>12} {'Std':>10} {'Success':>10} "
        f"{'EpLen':>8} {'ms/ep':>10}"
    )
    print("-" * 78)
    for row in rows:
        mk = float(row["mean_makespan"])  # type: ignore[arg-type]
        std = float(row["std_makespan"])  # type: ignore[arg-type]
        std_s = f"{std:.3f}" if math.isfinite(std) else "nan"
        print(
            f"{str(row['method']):<16} {_fmt_mk(mk):>12} {std_s:>10} "
            f"{float(row['success_rate']):10.1%} "
            f"{float(row['mean_ep_length']):8.1f} "
            f"{_episode_time_s(row) * 1000:10.3f}"
        )
    print("=" * 78)
    print(
        "ms/ep is mean wall time per instance. Constructive methods: "
        "mean step latency x episode length. MILP/LP: solver time per instance."
    )


def _row(
    name: str,
    result: EvalResult,
    *,
    time_is_per_episode: bool = False,
) -> Dict[str, object]:
    return {
        "method": name,
        "mean_makespan": float(result.mean_makespan),
        "std_makespan": float(result.std_makespan),
        "success_rate": float(result.success_rate),
        "mean_ep_length": float(result.mean_ep_length),
        "mean_inference_time_s": float(result.mean_inference_time_s),
        "time_is_per_episode": bool(time_is_per_episode),
        "n_success": int(result.n_success),
        "n_episodes": int(result.n_episodes),
    }


def _lp_rule_row(
    rule: str,
    makespans: List[float],
    template: EvalResult,
) -> Dict[str, object]:
    """One reconstruction rule, even when it loses to the other rule."""
    finite = [float(m) for m in makespans if math.isfinite(m)]
    n = len(makespans)
    n_success = len(finite)
    mean = float(sum(finite) / n_success) if finite else float("inf")
    if not finite:
        std = float("nan")
    elif n_success == 1:
        std = 0.0
    else:
        std = math.sqrt(sum((x - mean) ** 2 for x in finite) / n_success)
    return {
        "method": f"LP-{rule}",
        "mean_makespan": mean,
        "std_makespan": std,
        "success_rate": (n_success / n) if n else 0.0,
        "mean_ep_length": float(template.mean_ep_length),
        "mean_inference_time_s": float(template.mean_inference_time_s),
        "time_is_per_episode": True,
        "n_success": n_success,
        "n_episodes": n,
    }


def run_comparison(
    cfg: Optional[EvalConfig] = None,
    args: Optional[argparse.Namespace] = None,
) -> Tuple[List[Dict[str, object]], EvalConfig]:
    """Load PPO and evaluate every baseline on the same held-out stream."""
    explicit_path = bool(args is not None and getattr(args, "model_path", None) is not None)
    cfg = prepare_eval_config(cfg, args, apply_fn=apply_compare_args)
    model_path = resolve_model_path(cfg, explicit=explicit_path)
    trust = bool(getattr(args, "trust_checkpoint", False)) if args is not None else False
    if not trust:
        raise ValueError(
            f"Refusing to load SB3 checkpoint {model_path} without --trust-checkpoint."
        )
    rounding_trials = int(getattr(args, "rounding_trials", 20) if args is not None else 20)
    time_limit = getattr(args, "time_limit", None) if args is not None else None
    device = get_device(cfg.device)
    eval_env = make_eval_vec_env(cfg)
    rows: List[Dict[str, object]] = []

    try:
        logger.info("Loading model from %s (device=%s)", model_path, device)
        model = load_graph_ppo(model_path, eval_env, device)
        set_global_seed(cfg.seed, deterministic=True)

        eval_env.seed(cfg.seed)
        ppo = evaluate_policy_fjsp(
            model,
            eval_env,
            n_episodes=cfg.n_episodes,
            deterministic=cfg.deterministic,
        )
        rows.append(_row("PPO", ppo))

        eval_env.seed(cfg.seed)
        random = evaluate_random_fjsp(
            eval_env, n_episodes=cfg.n_episodes, seed=cfg.seed
        )
        rows.append(_row("Random", random))

        for rule in sorted(RULES):
            eval_env.seed(cfg.seed)
            heur = evaluate_heuristic_fjsp(
                eval_env, rule=rule, n_episodes=cfg.n_episodes
            )
            rows.append(_row(f"H-{rule}", heur))

        skip_milp = bool(getattr(args, "skip_milp", False)) if args is not None else False
        if not skip_milp:
            eval_env.seed(cfg.seed)
            milp = evaluate_milp_fjsp(
                eval_env,
                n_episodes=cfg.n_episodes,
                seed=cfg.seed,
                time_limit=time_limit,
            )
            rows.append(_row("MILP", milp, time_is_per_episode=True))

        eval_env.seed(cfg.seed)
        lp, lp_rows = evaluate_lp_rounding_fjsp(
            eval_env,
            n_episodes=cfg.n_episodes,
            seed=cfg.seed,
            rounding_trials=rounding_trials,
            time_limit=time_limit,
        )
        for rule in LIST_RULES:
            makespans = [
                float(rec.rule_makespans[rule])
                if rule in rec.rule_makespans
                else float("inf")
                for rec in lp_rows
            ]
            rows.append(_lp_rule_row(rule, makespans, lp))
    finally:
        eval_env.close()

    finite = [r for r in rows if math.isfinite(float(r["mean_makespan"]))]  # type: ignore[arg-type]
    finite.sort(key=lambda r: float(r["mean_makespan"]))  # type: ignore[arg-type]
    inf = [r for r in rows if not math.isfinite(float(r["mean_makespan"]))]  # type: ignore[arg-type]
    ordered = finite + inf
    _print_table(ordered)
    print(
        f"seed={cfg.seed}  n_episodes={cfg.n_episodes}  "
        f"env={cfg.env.n_machines}x{cfg.env.n_jobs}x{cfg.env.avg_operations_per_job}  "
        f"model={model_path}"
    )
    return ordered, cfg


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    run_comparison(eval_config_from_args(args), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

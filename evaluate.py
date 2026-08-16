"""Evaluate a trained FJSP PPO checkpoint.

Usage:
    python evaluate.py
    python evaluate.py --model-path ./checkpoints/best_model.zip --n-episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from config import EvalConfig
from models.graph_ppo import load_graph_ppo
from training.eval_cli import (
    add_shared_eval_args,
    apply_shared_eval_args,
    eval_config_from_args,
    make_eval_vec_env,
    prepare_eval_config,
)
from training.evaluate import evaluate_policy_fjsp, print_eval_result
from utils import (
    checkpoint_exists,
    get_device,
    get_logger,
    set_global_seed,
)

logger = get_logger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate FJSP PPO checkpoint")
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
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help=(
            "Required to load an SB3/cloudpickle checkpoint ZIP. These files are "
            "executable input and are not safe to load from untrusted sources."
        ),
    )
    add_shared_eval_args(parser)
    return parser.parse_args(argv)


def apply_args(cfg: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    """Apply CLI overrides to evaluation config."""
    cfg = apply_shared_eval_args(cfg, args)
    if args.model_path is not None:
        cfg.model_path = str(args.model_path)
    if args.device is not None:
        cfg.device = str(args.device)
    if args.stochastic:
        cfg.deterministic = False
    cfg.validate()
    return cfg


def resolve_model_path(cfg: EvalConfig, *, explicit: bool = False) -> Path:
    """Resolve the checkpoint path.

    Explicit CLI paths never fall back. A configured ``best_model.zip`` may fall
    back to a sibling ``latest_model.zip`` in the same directory.
    """
    path = Path(cfg.model_path)
    if checkpoint_exists(path):
        return path
    if explicit:
        raise FileNotFoundError(f"No checkpoint found at explicit path {path}.")

    latest = path.with_name("latest_model.zip")
    if path.name == "best_model.zip" and checkpoint_exists(latest):
        logger.warning(
            "Model %s not found; falling back to sibling %s",
            path,
            latest,
        )
        return latest

    raise FileNotFoundError(
        f"No checkpoint found at {path}. Train a model first with train.py."
    )


def evaluate(cfg: Optional[EvalConfig] = None, args: Optional[argparse.Namespace] = None):
    """Load a checkpoint, run deterministic evaluation, and print metrics."""
    explicit_path = bool(args is not None and args.model_path is not None)
    cfg = prepare_eval_config(cfg, args, apply_fn=apply_args)
    model_path = resolve_model_path(cfg, explicit=explicit_path)
    device = get_device(cfg.device)

    trust = bool(getattr(args, "trust_checkpoint", False)) if args is not None else False
    if not trust:
        raise ValueError(
            f"Refusing to load SB3 checkpoint {model_path} without --trust-checkpoint. "
            "SB3/cloudpickle ZIPs are executable input; only trust files you created."
        )

    eval_env = make_eval_vec_env(cfg)

    try:
        logger.info("Loading model from %s (device=%s)", model_path, device)
        model = load_graph_ppo(model_path, eval_env, device)
        # Re-apply CLI/eval seed after load so evaluation is comparable across runs.
        set_global_seed(cfg.seed, deterministic=True)
        eval_env.seed(cfg.seed)

        result = evaluate_policy_fjsp(
            model,
            eval_env,
            n_episodes=cfg.n_episodes,
            deterministic=cfg.deterministic,
        )
        print_eval_result(result)
        return result
    finally:
        eval_env.close()


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    evaluate(eval_config_from_args(args), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

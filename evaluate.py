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

from config import EvalConfig, get_default_eval_config, get_full_scale_eval_config
from models.graph_ppo import GraphPPO
from models.sb3_policy import GraphActorCriticPolicy
from training.eval_cli import (
    add_shared_eval_args,
    apply_shared_eval_args,
    build_eval_train_config,
)
from training.evaluate import evaluate_policy_fjsp, print_eval_result
from training.graph_buffer import GraphDictRolloutBuffer
from training.make_env import make_vec_env
from utils import (
    checkpoint_exists,
    configure_root_logging,
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
    return cfg


def resolve_model_path(cfg: EvalConfig, *, explicit: bool = False) -> Path:
    """Resolve the checkpoint path.

    Explicit CLI paths never fall back. The default configured path may fall
    back to a sibling ``latest_model.zip`` when the default best checkpoint is
    missing.
    """
    path = Path(cfg.model_path)
    if checkpoint_exists(path):
        return path
    if explicit:
        raise FileNotFoundError(f"No checkpoint found at explicit path {path}.")

    default_best = Path("./checkpoints/best_model.zip")
    if path.resolve() != default_best.resolve() and str(path) != str(default_best):
        # Non-default configured path: do not silently substitute another zip.
        raise FileNotFoundError(
            f"No checkpoint found at {path}. Train a model first with train.py."
        )

    latest = Path("./checkpoints/latest_model.zip")
    if checkpoint_exists(latest):
        logger.warning(
            "Model %s not found; falling back to default sibling %s",
            path,
            latest,
        )
        return latest

    raise FileNotFoundError(
        f"No checkpoint found at {path} or {latest}. Train a model first with train.py."
    )


def evaluate(cfg: Optional[EvalConfig] = None, args: Optional[argparse.Namespace] = None):
    """Load a checkpoint, run deterministic evaluation, and print metrics."""
    configure_root_logging()
    cfg = cfg or get_default_eval_config()
    explicit_path = False
    if args is not None:
        explicit_path = args.model_path is not None
        cfg = apply_args(cfg, args)
        cfg.validate()

    set_global_seed(cfg.seed, deterministic=True)
    model_path = resolve_model_path(cfg, explicit=explicit_path)
    device = get_device(cfg.device)

    trust = bool(getattr(args, "trust_checkpoint", False)) if args is not None else False
    if not trust:
        raise ValueError(
            f"Refusing to load SB3 checkpoint {model_path} without --trust-checkpoint. "
            "SB3/cloudpickle ZIPs are executable input; only trust files you created."
        )

    train_cfg = build_eval_train_config(cfg)
    eval_env = make_vec_env(
        train_cfg,
        n_envs=1,
        use_subprocess=False,
        for_eval=True,
    )

    try:
        logger.info("Loading model from %s (device=%s)", model_path, device)
        model = GraphPPO.load(
            str(model_path),
            env=eval_env,
            device=device,
            custom_objects={
                "policy_class": GraphActorCriticPolicy,
                "rollout_buffer_class": GraphDictRolloutBuffer,
            },
        )
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
    cfg = (
        get_full_scale_eval_config()
        if getattr(args, "full_scale", False)
        else get_default_eval_config()
    )
    evaluate(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Train FJSP PPO with Stable-Baselines3 and a graph policy.

Usage:
    python train.py
    python train.py --n-envs 8 --total-timesteps 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from callbacks import build_callbacks, make_lr_schedule
from config import TrainConfig, get_default_train_config
from models.graph_ppo import GraphPPO
from models.sb3_policy import GraphActorCriticPolicy, make_policy_kwargs
from training.graph_buffer import GraphDictRolloutBuffer
from training.make_env import make_vec_env
from utils import (
    checkpoint_exists,
    configure_root_logging,
    ensure_dir,
    get_device,
    get_logger,
    set_global_seed,
)

logger = get_logger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments that override ``TrainConfig`` defaults."""
    parser = argparse.ArgumentParser(description="Train FJSP PPO")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None, choices=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto | cpu | cuda | cuda:0")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly resume from latest checkpoint (default: start fresh)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing latest checkpoint (default behavior)",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help=(
            "Required to load an SB3/cloudpickle checkpoint ZIP. These files are "
            "executable input and are not safe to load from untrusted sources."
        ),
    )
    parser.add_argument("--dummy-vec", action="store_true", help="Force in-process GraphDummyVecEnv")
    parser.add_argument(
        "--n-machines",
        type=int,
        default=None,
        help="Override env n_machines (default from config)",
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--avg-ops", type=int, default=None, dest="avg_operations_per_job")
    return parser.parse_args(argv)


def apply_args(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Apply CLI overrides onto a config instance and revalidate."""
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.n_envs is not None:
        cfg.n_envs = int(args.n_envs)
    if args.total_timesteps is not None:
        cfg.ppo.total_timesteps = int(args.total_timesteps)
    if args.device is not None:
        cfg.device = str(args.device)
    if getattr(args, "resume", False):
        cfg.resume = True
    if getattr(args, "no_resume", False):
        cfg.resume = False
    if getattr(args, "trust_checkpoint", False):
        cfg.trust_checkpoint = True
    if args.n_machines is not None:
        cfg.env.n_machines = int(args.n_machines)
    if args.n_jobs is not None:
        cfg.env.n_jobs = int(args.n_jobs)
    if args.avg_operations_per_job is not None:
        cfg.env.avg_operations_per_job = int(args.avg_operations_per_job)
    cfg.validate()
    return cfg


def build_ppo(cfg: TrainConfig, train_env) -> GraphPPO:
    """Create a fresh GraphPPO model with ``GraphActorCriticPolicy``."""
    device = get_device(cfg.device)
    learning_rate = make_lr_schedule(cfg)
    policy_kwargs = make_policy_kwargs(cfg.model)

    rollout_size = int(cfg.ppo.n_steps) * int(cfg.n_envs)
    if rollout_size % int(cfg.ppo.batch_size) != 0:
        raise ValueError(
            f"n_steps*n_envs ({rollout_size}) must be divisible by "
            f"batch_size ({cfg.ppo.batch_size})"
        )

    model = GraphPPO(
        policy=GraphActorCriticPolicy,
        env=train_env,
        learning_rate=learning_rate,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        target_kl=cfg.ppo.target_kl,
        tensorboard_log=cfg.tensorboard_log,
        policy_kwargs=policy_kwargs,
        rollout_buffer_class=GraphDictRolloutBuffer,
        seed=cfg.seed,
        device=device,
        verbose=1,
    )
    logger.info(
        "Created GraphPPO on device=%s n_envs=%d n_steps=%d batch_size=%d",
        device,
        cfg.n_envs,
        cfg.ppo.n_steps,
        cfg.ppo.batch_size,
    )
    return model


def resolve_resume_path(cfg: TrainConfig) -> Optional[Path]:
    """Return the latest checkpoint path when resume is enabled and trusted."""
    latest = cfg.latest_model_path()
    if not (cfg.resume and checkpoint_exists(latest)):
        return None
    if not cfg.trust_checkpoint:
        raise ValueError(
            f"Refusing to load SB3 checkpoint {latest} without --trust-checkpoint. "
            "SB3/cloudpickle ZIPs are executable input; only trust files you created."
        )
    from training.checkpoints import assert_config_compatible

    assert_config_compatible(latest, cfg.to_dict(), require_metadata=True)
    return latest


def maybe_resume(cfg: TrainConfig, train_env) -> GraphPPO:
    """Load ``latest_model.zip`` when resume is explicitly enabled and trusted."""
    latest = resolve_resume_path(cfg)
    if latest is None:
        return build_ppo(cfg, train_env)

    logger.info("Resuming training from %s", latest)
    device = get_device(cfg.device)
    try:
        model = GraphPPO.load(
            str(latest),
            env=train_env,
            device=device,
            custom_objects={
                "policy_class": GraphActorCriticPolicy,
                "rollout_buffer_class": GraphDictRolloutBuffer,
            },
        )
    except Exception:
        # Preserve the original load error; do not wrap away the root cause.
        raise
    # Restore schedule with progress consistent with already-consumed timesteps.
    model.learning_rate = make_lr_schedule(cfg)
    model._setup_lr_schedule()
    return model


def train(cfg: Optional[TrainConfig] = None, args: Optional[argparse.Namespace] = None) -> GraphPPO:
    """Run the full training loop and return the trained model."""
    configure_root_logging()
    cfg = cfg or get_default_train_config()
    if args is not None:
        cfg = apply_args(cfg, args)

    ensure_dir(cfg.checkpoint_dir)
    ensure_dir(cfg.tensorboard_log)
    set_global_seed(cfg.seed, deterministic=cfg.deterministic_torch)

    use_subprocess = True
    if args is not None and args.dummy_vec:
        use_subprocess = False
    if cfg.n_envs == 1:
        use_subprocess = False

    logger.info(
        "Building train env (n_envs=%d subprocess=%s) instance=%dx%dx%d",
        cfg.n_envs,
        use_subprocess,
        cfg.env.n_machines,
        cfg.env.n_jobs,
        cfg.env.avg_operations_per_job,
    )
    train_env = make_vec_env(
        cfg,
        n_envs=cfg.n_envs,
        use_subprocess=use_subprocess,
        monitor_dir=str(Path(cfg.tensorboard_log) / "monitor_train"),
        for_eval=False,
    )
    eval_env = make_vec_env(
        cfg,
        n_envs=1,
        use_subprocess=False,
        monitor_dir=str(Path(cfg.tensorboard_log) / "monitor_eval"),
        for_eval=True,
    )

    try:
        model = maybe_resume(cfg, train_env)
        callbacks = build_callbacks(cfg, eval_env=eval_env, verbose=1)

        logger.info("Starting learn(total_timesteps=%d)", cfg.ppo.total_timesteps)
        model.learn(
            total_timesteps=int(cfg.ppo.total_timesteps),
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not (cfg.resume and checkpoint_exists(cfg.latest_model_path())),
        )

        from training.checkpoints import write_checkpoint_metadata

        final_path = cfg.latest_model_path()
        model.save(str(final_path))
        write_checkpoint_metadata(final_path, config=cfg.to_dict(), extra={"final": True})
        logger.info("Training complete. Latest model saved to %s", final_path)
        return model
    finally:
        train_env.close()
        eval_env.close()


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    train(get_default_train_config(), args)
    return 0


if __name__ == "__main__":
    # Required on Windows for SubprocVecEnv spawn.
    sys.exit(main())

"""Training callbacks for FJSP PPO: checkpoints, eval, TensorBoard, LR schedule."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnv

from config import TrainConfig
from training.evaluate import evaluate_policy_fjsp
from utils import ensure_dir, get_logger

logger = get_logger(__name__)


def linear_schedule(
    initial_value: float,
    end_fraction: float = 0.1,
) -> Callable[[float], float]:
    """Linear learning-rate schedule for SB3 (``progress_remaining`` in [1, 0]).

    Decays from ``initial_value`` down to ``initial_value * end_fraction``
    (not to zero), so late training keeps a non-zero learning rate.
    """
    end_fraction = float(end_fraction)
    if not 0.0 <= end_fraction <= 1.0:
        raise ValueError(f"end_fraction must be in [0, 1], got {end_fraction}")

    def schedule(progress_remaining: float) -> float:
        # progress_remaining: 1 at start -> 0 at end
        return float(
            initial_value
            * (end_fraction + (1.0 - end_fraction) * progress_remaining)
        )

    return schedule


def constant_schedule(initial_value: float) -> Callable[[float], float]:
    """Constant learning-rate schedule."""

    def schedule(progress_remaining: float) -> float:
        return float(initial_value)

    return schedule


def make_lr_schedule(cfg: TrainConfig) -> Union[float, Callable[[float], float]]:
    """Build the SB3 learning-rate argument from config."""
    lr = float(cfg.ppo.learning_rate)
    if cfg.lr_schedule == "linear":
        return linear_schedule(lr, end_fraction=float(cfg.lr_end_fraction))
    return constant_schedule(lr)


class LatestCheckpointCallback(BaseCallback):
    """Save ``latest_model.zip`` every N completed PPO updates."""

    def __init__(
        self,
        save_path: Union[str, Path],
        save_freq_updates: int,
        n_steps: int,
        n_envs: int,
        verbose: int = 0,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_freq_updates = max(1, int(save_freq_updates))
        self.timesteps_per_update = int(n_steps) * int(n_envs)
        self._last_saved_update = -1
        self.config = config

    def _on_training_start(self) -> None:
        ensure_dir(self.save_path.parent)

    def _maybe_save(self) -> None:
        if self.timesteps_per_update <= 0:
            return
        if self.num_timesteps % self.timesteps_per_update != 0:
            return
        update_idx = self.num_timesteps // self.timesteps_per_update
        if update_idx <= 0 or update_idx % self.save_freq_updates != 0:
            return
        if update_idx == self._last_saved_update:
            return
        from training.checkpoints import save_checkpoint

        save_checkpoint(
            self.model,
            self.save_path,
            config=self.config,
            extra={"num_timesteps": int(self.num_timesteps), "update_idx": int(update_idx)},
        )
        self._last_saved_update = update_idx
        if self.verbose:
            logger.info(
                "Saved latest checkpoint at update %d -> %s",
                update_idx,
                self.save_path,
            )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._maybe_save()

    def _on_training_end(self) -> None:
        ensure_dir(self.save_path.parent)
        from training.checkpoints import save_checkpoint

        save_checkpoint(
            self.model,
            self.save_path,
            config=self.config,
            extra={"num_timesteps": int(self.num_timesteps), "final": True},
        )
        logger.info("Saved final latest checkpoint -> %s", self.save_path)


class BestModelCallback(BaseCallback):
    """Track the best evaluation score and save ``best_model.zip``."""

    def __init__(
        self,
        save_path: Union[str, Path],
        verbose: int = 0,
        config: Optional[dict] = None,
        metric: str = "mean_makespan",
    ) -> None:
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.metric = str(metric)
        if self.metric not in ("mean_reward", "mean_makespan"):
            raise ValueError(f"Unsupported best metric: {self.metric!r}")
        # Lower-is-better for makespan; higher-is-better for reward.
        self.best_score = np.inf if self.metric == "mean_makespan" else -np.inf
        self.config = config

    def load_persisted_best(self, checkpoint_dir: Union[str, Path]) -> None:
        """Preserve an existing best score across resume."""
        from training.checkpoints import load_best_score_record

        record = load_best_score_record(checkpoint_dir)
        if record is None:
            return
        if "best_score" not in record:
            return
        score = float(record["best_score"])
        if str(record.get("best_metric", self.metric)) != self.metric:
            logger.warning(
                "Ignoring persisted best score with metric=%s (expected %s)",
                record.get("best_metric"),
                self.metric,
            )
            return
        if self.config is not None:
            from training.checkpoints import config_fingerprint, meta_path_for

            if meta_path_for(self.save_path).is_file():
                from training.checkpoints import read_checkpoint_metadata

                meta = read_checkpoint_metadata(self.save_path)
                expected = config_fingerprint(self.config)
                actual = str(meta.get("config_fingerprint", ""))
                if actual and actual != expected:
                    logger.warning(
                        "Ignoring persisted best score; %s fingerprint != current config",
                        self.save_path.name,
                    )
                    return
        self.best_score = float(score)

    def update(self, score: float) -> bool:
        """Save when the configured metric improves. Returns True if improved."""
        value = float(score)
        if self.metric == "mean_makespan":
            if not np.isfinite(value):
                return False
            improved = value < float(self.best_score)
        else:
            improved = value > float(self.best_score)
        if not improved:
            return False

        self.best_score = value
        ensure_dir(self.save_path.parent)
        from training.checkpoints import save_best_score, save_checkpoint

        save_checkpoint(
            self.model,
            self.save_path,
            config=self.config,
            extra={
                "best_metric": self.metric,
                "best_score": self.best_score,
            },
        )
        save_best_score(self.save_path.parent, self.best_score, metric=self.metric)
        logger.info(
            "New best %s %.4f -> saved %s",
            self.metric,
            self.best_score,
            self.save_path,
        )
        return True

    def _on_step(self) -> bool:
        return True


class FJSPEvalCallback(BaseCallback):
    """Periodic evaluation with makespan / success logging and best-model saving."""

    def __init__(
        self,
        eval_env: VecEnv,
        best_model_path: Union[str, Path],
        n_eval_episodes: int,
        eval_freq_updates: int,
        n_steps: int,
        n_envs: int,
        deterministic: bool = True,
        verbose: int = 0,
        config: Optional[dict] = None,
        eval_seed: int = 0,
        best_metric: str = "mean_makespan",
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_freq_updates = max(1, int(eval_freq_updates))
        self.timesteps_per_update = int(n_steps) * int(n_envs)
        self.deterministic = bool(deterministic)
        self.eval_seed = int(eval_seed)
        self.best_callback = BestModelCallback(
            best_model_path,
            verbose=verbose,
            config=config,
            metric=best_metric,
        )
        self._last_eval_update = -1
        self.config = config

    def _on_training_start(self) -> None:
        self.best_callback.model = self.model
        self.best_callback.num_timesteps = self.num_timesteps
        self.best_callback.load_persisted_best(Path(self.best_callback.save_path).parent)

    def _maybe_eval(self, *, force: bool = False) -> None:
        if self.timesteps_per_update <= 0:
            return
        if not force and self.num_timesteps % self.timesteps_per_update != 0:
            return
        update_idx = self.num_timesteps // self.timesteps_per_update
        if not force and (update_idx <= 0 or update_idx % self.eval_freq_updates != 0):
            return
        if not force and update_idx == self._last_eval_update:
            return

        self._last_eval_update = update_idx
        # Held-out deterministic episode schedule: eval_seed, eval_seed+1, ...
        self.eval_env.seed(self.eval_seed)
        result = evaluate_policy_fjsp(
            self.model,
            self.eval_env,
            n_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
        )
        self.best_callback.model = self.model
        score = result.mean_makespan
        # Heuristics are 100% success; don't crown a partial-success makespan.
        if result.success_rate < 1.0:
            score = float("inf")
        self.best_callback.update(score)

        if self.logger is not None:
            self.logger.record("eval/mean_makespan", result.mean_makespan)
            self.logger.record("eval/mean_ep_length", result.mean_ep_length)
            self.logger.record("eval/success_rate", result.success_rate)
            self.logger.record(
                "eval/mean_inference_time_ms",
                result.mean_inference_time_s * 1000.0,
            )
        if self.verbose:
            logger.info("Eval at update %d:\n%s", update_idx, result.format_summary())

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._maybe_eval(force=False)

    def _on_training_end(self) -> None:
        self._maybe_eval(force=True)

class TensorboardCallback(BaseCallback):
    """Log extra FJSP metrics to TensorBoard / SB3 logger."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._ep_rewards: List[float] = []
        self._ep_lengths: List[float] = []
        self._ep_makespans: List[float] = []
        self._ep_successes: List[float] = []
        self._last_time = time.time()
        self._last_timesteps = 0

    def _on_step(self) -> bool:
        # Cheap path: only collect episode stats every env step.
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                if not isinstance(info, dict):
                    continue
                episode = info.get("episode")
                if episode is None and isinstance(info.get("final_info"), dict):
                    episode = info["final_info"].get("episode")
                if not isinstance(episode, dict):
                    continue
                self._ep_rewards.append(float(episode.get("r", 0.0)))
                self._ep_lengths.append(float(episode.get("l", 0.0)))
                self._ep_makespans.append(float(episode.get("makespan", float("inf"))))
                self._ep_successes.append(1.0 if bool(episode.get("success", False)) else 0.0)
        return True

    def _on_rollout_end(self) -> None:
        self._log_metrics()

    def _log_metrics(self) -> None:
        if self.logger is None:
            return

        now = time.time()
        dt = max(now - self._last_time, 1e-8)
        dsteps = max(self.num_timesteps - self._last_timesteps, 0)
        fps = dsteps / dt
        self._last_time = now
        self._last_timesteps = self.num_timesteps

        if self._ep_rewards:
            window = min(len(self._ep_rewards), 50)
            self.logger.record(
                "rollout/ep_rew_mean_fjsp",
                float(np.mean(self._ep_rewards[-window:])),
            )
            self.logger.record(
                "rollout/ep_len_mean_fjsp",
                float(np.mean(self._ep_lengths[-window:])),
            )
            finite = [m for m in self._ep_makespans[-window:] if np.isfinite(m)]
            if finite:
                self.logger.record("rollout/makespan_mean", float(np.mean(finite)))
            self.logger.record(
                "rollout/success_rate",
                float(np.mean(self._ep_successes[-window:])),
            )

        if hasattr(self.model, "lr_schedule") and hasattr(
            self.model, "_current_progress_remaining"
        ):
            try:
                lr = float(self.model.lr_schedule(self.model._current_progress_remaining))
                self.logger.record("train/learning_rate_fjsp", lr)
            except (TypeError, ValueError, AttributeError):
                pass

        grad_norm = self._compute_grad_norm()
        if grad_norm is not None:
            self.logger.record("train/grad_norm", grad_norm)

        self.logger.record("time/fps_fjsp", float(fps))

    def _compute_grad_norm(self) -> Optional[float]:
        policy = getattr(self.model, "policy", None)
        if policy is None:
            return None
        total = 0.0
        found = False
        for param in policy.parameters():
            grad = param.grad
            if grad is None:
                continue
            total += float(grad.data.norm(2).item() ** 2)
            found = True
        if not found:
            return None
        return float(total ** 0.5)


class LearningRateSchedulerCallback(BaseCallback):
    """Explicitly record / sync learning rate for TensorBoard visibility.

    SB3 already applies callable schedules; this callback mirrors the current LR
    into the logger each step for reliable dashboards.
    """

    def _on_step(self) -> bool:
        if self.logger is None:
            return True
        optimizer = getattr(self.model.policy, "optimizer", None)
        if optimizer is None:
            return True
        lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if lrs:
            self.logger.record("train/learning_rate", float(np.mean(lrs)))
        return True


def build_callbacks(
    cfg: TrainConfig,
    eval_env: VecEnv,
    verbose: int = 1,
) -> CallbackList:
    """Assemble the full callback stack from config."""
    ensure_dir(cfg.checkpoint_dir)
    ensure_dir(cfg.tensorboard_log)
    config_dict = cfg.to_dict()

    latest_cb = LatestCheckpointCallback(
        save_path=cfg.latest_model_path(),
        save_freq_updates=cfg.checkpoint_freq_updates,
        n_steps=cfg.ppo.n_steps,
        n_envs=cfg.n_envs,
        verbose=verbose,
        config=config_dict,
    )
    eval_cb = FJSPEvalCallback(
        eval_env=eval_env,
        best_model_path=cfg.best_model_path(),
        n_eval_episodes=cfg.n_eval_episodes,
        eval_freq_updates=cfg.eval_freq_updates,
        n_steps=cfg.ppo.n_steps,
        n_envs=cfg.n_envs,
        deterministic=True,
        verbose=verbose,
        config=config_dict,
        eval_seed=int(cfg.eval_seed),
        best_metric=str(cfg.best_metric),
    )
    tb_cb = TensorboardCallback(verbose=verbose)
    lr_cb = LearningRateSchedulerCallback()

    callbacks = CallbackList([latest_cb, eval_cb, tb_cb, lr_cb])
    logger.info(
        "Built callbacks: checkpoint every %d updates, eval every %d updates",
        cfg.checkpoint_freq_updates,
        cfg.eval_freq_updates,
    )
    return callbacks


__all__ = [
    "BestModelCallback",
    "FJSPEvalCallback",
    "LatestCheckpointCallback",
    "LearningRateSchedulerCallback",
    "TensorboardCallback",
    "build_callbacks",
    "constant_schedule",
    "linear_schedule",
    "make_lr_schedule",
]

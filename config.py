"""Central configuration for FJSP PPO training and evaluation.

All hyperparameters live here. Other modules must import from this file
instead of hard-coding constants.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional

_DEVICE_RE = re.compile(r"^(?:auto|cpu|mps|cuda(?::(\d+))?)$")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _require_probability(name: str, value: float) -> None:
    try:
        value_f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}") from exc
    if not 0.0 <= value_f <= 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")


def _require_positive_float(name: str, value: float) -> None:
    try:
        value_f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive float, got {value!r}") from exc
    if value_f <= 0.0:
        raise ValueError(f"{name} must be a positive float, got {value!r}")


def validate_device(device: str) -> None:
    """Accept only ``auto``, ``cpu``, ``mps``, ``cuda``, or ``cuda:N`` with N >= 0."""
    if not isinstance(device, str):
        raise ValueError(f"device must be a string, got {device!r}")
    match = _DEVICE_RE.fullmatch(device)
    if match is None:
        raise ValueError(
            f"device must be 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N' (N>=0), "
            f"got {device!r}"
        )
    if match.group(1) is not None and int(match.group(1)) < 0:
        raise ValueError(f"device CUDA index must be >= 0, got {device!r}")


@dataclass
class EnvConfig:
    """FJSP environment instance parameters."""

    n_machines: int = 5
    n_jobs: int = 3
    avg_operations_per_job: int = 4
    time_penalty: float = -0.1
    max_operation_duration: int = 20
    connection_drop_prob: float = 0.6
    compatible_efficiency_std: float = 0.2
    time_step: float = 1.0
    min_eligible_machines: int = 2
    cross_job_dep_prob: float = 0.6
    shared_dep_prob: float = 0.4

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n_operations(self) -> int:
        """Expected operation count used for action-space sizing."""
        return int(self.n_jobs * self.avg_operations_per_job)

    @property
    def n_actions(self) -> int:
        """Flattened discrete action-space size."""
        return self.n_machines * self.n_operations

    def validate(self) -> None:
        """Raise ``ValueError`` when environment parameters are inconsistent."""
        _require_positive_int("n_machines", self.n_machines)
        _require_positive_int("n_jobs", self.n_jobs)
        _require_positive_int("avg_operations_per_job", self.avg_operations_per_job)
        _require_positive_int("max_operation_duration", self.max_operation_duration)
        _require_positive_int("min_eligible_machines", self.min_eligible_machines)
        if self.min_eligible_machines > self.n_machines:
            raise ValueError(
                "min_eligible_machines must be <= n_machines, "
                f"got {self.min_eligible_machines} > {self.n_machines}"
            )
        _require_probability("connection_drop_prob", self.connection_drop_prob)
        _require_probability("cross_job_dep_prob", self.cross_job_dep_prob)
        _require_probability("shared_dep_prob", self.shared_dep_prob)
        _require_positive_float("time_step", self.time_step)
        try:
            std = float(self.compatible_efficiency_std)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"compatible_efficiency_std must be >= 0, got {self.compatible_efficiency_std!r}"
            ) from exc
        if std < 0.0:
            raise ValueError(
                f"compatible_efficiency_std must be >= 0, got {self.compatible_efficiency_std!r}"
            )


@dataclass
class ModelConfig:
    """Graph encoder and actor-critic hyperparameters."""

    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.0
    predictor_type: Literal["dot_product", "bilinear", "attention"] = "bilinear"
    operation_in_dim: int = 10
    machine_in_dim: int = 3
    critic_hidden_dim: int = 256

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` when model dimensions or dropout are invalid."""
        for name in (
            "hidden_dim",
            "num_layers",
            "num_heads",
            "operation_in_dim",
            "machine_in_dim",
            "critic_hidden_dim",
        ):
            _require_positive_int(name, getattr(self, name))
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if float(self.dropout) != 0.0:
            raise ValueError(
                "dropout must be 0.0 for deterministic PPO likelihoods, "
                f"got {self.dropout!r}"
            )
        if self.predictor_type not in ("dot_product", "bilinear", "attention"):
            raise ValueError(f"predictor_type is invalid: {self.predictor_type!r}")


@dataclass
class PPOConfig:
    """Stable-Baselines3 PPO hyperparameters."""

    learning_rate: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 128
    n_epochs: int = 4
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 1_000_000
    # Early-stop PPO epochs when approx KL exceeds this (None disables).
    target_kl: float = 0.05

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` when PPO batch/update values are invalid."""
        _require_positive_int("n_steps", self.n_steps)
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_int("n_epochs", self.n_epochs)
        _require_positive_int("total_timesteps", self.total_timesteps)
        _require_positive_float("learning_rate", self.learning_rate)
        _require_positive_float("max_grad_norm", self.max_grad_norm)
        for name in ("gamma", "gae_lambda", "clip_range"):
            _require_probability(name, getattr(self, name))
        try:
            ent = float(self.ent_coef)
            vf = float(self.vf_coef)
        except (TypeError, ValueError) as exc:
            raise ValueError("ent_coef and vf_coef must be finite floats") from exc
        if ent < 0.0 or vf < 0.0:
            raise ValueError("ent_coef and vf_coef must be >= 0")
        if self.target_kl is not None:
            _require_positive_float("target_kl", float(self.target_kl))


# Held-out eval seeds sit far from training worker seeds (seed + rank*1000).
EVAL_SEED_OFFSET = 1_000_000


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    seed: int = 42
    n_envs: int = 8
    deterministic_torch: bool = True
    device: str = "auto"
    tensorboard_log: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    latest_model_name: str = "latest_model.zip"
    best_model_name: str = "best_model.zip"
    checkpoint_freq_updates: int = 10
    eval_freq_updates: int = 10
    n_eval_episodes: int = 5
    # Held-out eval seed base; episode i uses eval_seed + i.
    eval_seed: Optional[int] = None
    best_metric: Literal["mean_reward", "mean_makespan"] = "mean_makespan"
    lr_schedule: Literal["constant", "linear"] = "linear"
    lr_end_fraction: float = 0.1
    normalize_reward: bool = False
    resume: bool = False
    trust_checkpoint: bool = False
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.env, EnvConfig):
            self.env = EnvConfig(**self.env)  # type: ignore[arg-type]
        if not isinstance(self.model, ModelConfig):
            self.model = ModelConfig(**self.model)  # type: ignore[arg-type]
        if not isinstance(self.ppo, PPOConfig):
            self.ppo = PPOConfig(**self.ppo)  # type: ignore[arg-type]
        if self.eval_seed is None:
            self.eval_seed = int(self.seed) + EVAL_SEED_OFFSET
        self.validate()

    def latest_model_path(self) -> Path:
        """Path to the latest training checkpoint."""
        return Path(self.checkpoint_dir) / self.latest_model_name

    def best_model_path(self) -> Path:
        """Path to the best evaluation checkpoint."""
        return Path(self.checkpoint_dir) / self.best_model_name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain dictionary."""
        return asdict(self)

    def validate(self) -> None:
        """Raise ``ValueError`` when train/eval counts or PPO batching are invalid."""
        _require_non_negative_int("seed", self.seed)
        _require_positive_int("n_envs", self.n_envs)
        _require_positive_int("checkpoint_freq_updates", self.checkpoint_freq_updates)
        _require_positive_int("eval_freq_updates", self.eval_freq_updates)
        _require_positive_int("n_eval_episodes", self.n_eval_episodes)
        _require_non_negative_int("eval_seed", int(self.eval_seed))
        if self.best_metric not in ("mean_reward", "mean_makespan"):
            raise ValueError(f"best_metric is invalid: {self.best_metric!r}")
        validate_device(self.device)
        if self.lr_schedule not in ("constant", "linear"):
            raise ValueError(f"lr_schedule is invalid: {self.lr_schedule!r}")
        _require_probability("lr_end_fraction", self.lr_end_fraction)
        self.env.validate()
        self.model.validate()
        self.ppo.validate()
        rollout_size = int(self.ppo.n_steps) * int(self.n_envs)
        if rollout_size % int(self.ppo.batch_size) != 0:
            raise ValueError(
                f"n_steps*n_envs ({rollout_size}) must be divisible by "
                f"batch_size ({self.ppo.batch_size})"
            )


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    seed: int = 42
    n_episodes: int = 20
    deterministic: bool = True
    model_path: str = "./checkpoints/best_model.zip"
    device: str = "auto"
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.env, EnvConfig):
            self.env = EnvConfig(**self.env)  # type: ignore[arg-type]
        if not isinstance(self.model, ModelConfig):
            self.model = ModelConfig(**self.model)  # type: ignore[arg-type]
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` when evaluation parameters are invalid."""
        _require_non_negative_int("seed", self.seed)
        _require_positive_int("n_episodes", self.n_episodes)
        validate_device(self.device)
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError(f"model_path must be a non-empty string, got {self.model_path!r}")
        self.env.validate()
        self.model.validate()


# Demo / smoke-test instance (EnvConfig field defaults): 5×3×4.
DEBUG_SCALE_ENV = EnvConfig()

# Full-scale instance for later serious runs (switch get_default_* back to this).
FULL_SCALE_ENV = EnvConfig(
    n_machines=25,
    n_jobs=15,
    avg_operations_per_job=8,
)


def get_debug_train_config() -> TrainConfig:
    """Small demo config: tiny FJSP instance + short PPO run to verify training."""
    return TrainConfig(
        n_envs=2,
        checkpoint_freq_updates=4,
        eval_freq_updates=4,
        n_eval_episodes=5,
        lr_schedule="linear",
        lr_end_fraction=0.2,
        env=replace(DEBUG_SCALE_ENV),
        model=ModelConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            critic_hidden_dim=128,
        ),
        ppo=PPOConfig(
            learning_rate=1e-4,
            gamma=1.0,
            clip_range=0.2,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            ent_coef=0.05,
            vf_coef=0.5,
            target_kl=0.05,
            total_timesteps=32_768,
        ),
    )


def get_default_train_config() -> TrainConfig:
    """Return the default training configuration (demo-scale until verified)."""
    return get_debug_train_config()


def get_default_eval_config() -> EvalConfig:
    """Return the default evaluation configuration (demo-scale until verified)."""
    cfg = get_debug_train_config()
    return EvalConfig(env=replace(cfg.env), model=replace(cfg.model))

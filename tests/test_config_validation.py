from argparse import Namespace

import pytest

from config import (
    EnvConfig,
    EvalConfig,
    ModelConfig,
    PPOConfig,
    TrainConfig,
    get_default_eval_config,
    get_default_train_config,
)
from train import apply_args
from training.eval_cli import apply_shared_eval_args


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: EnvConfig(n_machines=0), "n_machines"),
        (lambda: EnvConfig(connection_drop_prob=1.1), "connection_drop_prob"),
        (lambda: EnvConfig(time_step=0.0), "time_step"),
        (
            lambda: EnvConfig(n_machines=2, min_eligible_machines=3),
            "min_eligible_machines",
        ),
        (lambda: ModelConfig(hidden_dim=0), "hidden_dim"),
        (lambda: ModelConfig(dropout=0.1), "dropout"),
        (lambda: PPOConfig(n_steps=0), "n_steps"),
        (lambda: PPOConfig(batch_size=0), "batch_size"),
        (lambda: PPOConfig(n_epochs=0), "n_epochs"),
        (lambda: TrainConfig(seed=-1), "seed"),
        (lambda: TrainConfig(n_envs=0), "n_envs"),
        (lambda: TrainConfig(device="gpu"), "device"),
        (lambda: TrainConfig(checkpoint_freq_updates=0), "checkpoint_freq_updates"),
        (lambda: EvalConfig(n_episodes=0), "n_episodes"),
        (lambda: EvalConfig(device="cuda:-1"), "device"),
    ],
)
def test_config_constructors_reject_invalid_values(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_default_configs_are_valid_and_ppo_batches_divide_rollout():
    train_cfg = get_default_train_config()
    eval_cfg = get_default_eval_config()

    train_cfg.validate()
    eval_cfg.validate()
    assert train_cfg.model.dropout == 0.0
    assert train_cfg.env.n_machines == 5
    assert train_cfg.env.n_jobs == 3
    assert (train_cfg.ppo.n_steps * train_cfg.n_envs) % train_cfg.ppo.batch_size == 0


@pytest.mark.parametrize("device", ["auto", "cpu", "mps", "cuda", "cuda:0"])
def test_device_strings_are_accepted(device):
    TrainConfig(device=device).validate()
    EvalConfig(device=device).validate()


def test_train_config_rejects_partial_ppo_minibatch():
    with pytest.raises(ValueError, match="n_steps\\*n_envs"):
        TrainConfig(n_envs=3, ppo=PPOConfig(n_steps=5, batch_size=4))


def test_train_cli_revalidates_after_mutation():
    args = Namespace(
        seed=None,
        n_envs=None,
        total_timesteps=None,
        device=None,
        resume=False,
        no_resume=False,
        trust_checkpoint=False,
        dummy_vec=False,
        n_machines=0,
        n_jobs=None,
        avg_operations_per_job=None,
    )

    with pytest.raises(ValueError, match="n_machines"):
        apply_args(get_default_train_config(), args)


def test_eval_cli_revalidates_after_mutation():
    args = Namespace(
        n_episodes=0,
        seed=None,
        n_machines=None,
        n_jobs=None,
        avg_operations_per_job=None,
    )

    with pytest.raises(ValueError, match="n_episodes"):
        apply_shared_eval_args(get_default_eval_config(), args)


def test_direct_env_constructor_reuses_config_validation():
    from envs.fjsp_env import FJSPEnv

    with pytest.raises(ValueError, match="min_eligible_machines"):
        FJSPEnv(n_machines=1, min_eligible_machines=2)

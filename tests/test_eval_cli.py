from argparse import Namespace

from config import get_default_eval_config
from training.eval_cli import apply_shared_eval_args, build_eval_train_config


def test_apply_shared_eval_args_overrides_env_and_episodes():
    cfg = get_default_eval_config()
    args = Namespace(
        n_episodes=3,
        seed=7,
        n_machines=4,
        n_jobs=2,
        avg_operations_per_job=3,
    )
    cfg = apply_shared_eval_args(cfg, args)
    assert cfg.n_episodes == 3
    assert cfg.seed == 7
    assert cfg.env.n_machines == 4
    assert cfg.env.n_jobs == 2
    assert cfg.env.avg_operations_per_job == 3


def test_build_eval_train_config_disables_reward_norm():
    cfg = get_default_eval_config()
    cfg.seed = 11
    train_cfg = build_eval_train_config(cfg)
    assert train_cfg.n_envs == 1
    assert train_cfg.normalize_reward is False
    assert train_cfg.seed == 11
    assert train_cfg.env.n_machines == cfg.env.n_machines

from argparse import Namespace

from config import get_default_eval_config, get_full_scale_eval_config
from training.eval_cli import (
    apply_shared_eval_args,
    build_eval_train_config,
    eval_config_from_args,
)


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


def test_apply_shared_eval_args_full_scale_upgrades_demo_config():
    demo = get_default_eval_config()
    full = get_full_scale_eval_config()
    args = Namespace(full_scale=True)
    cfg = apply_shared_eval_args(demo, args)
    assert cfg.env.n_machines == full.env.n_machines
    assert cfg.env.n_jobs == full.env.n_jobs
    assert cfg.env.avg_operations_per_job == full.env.avg_operations_per_job
    assert cfg.model.hidden_dim == full.model.hidden_dim
    assert cfg.model.num_layers == full.model.num_layers


def test_apply_shared_eval_args_full_scale_then_size_overrides():
    args = Namespace(full_scale=True, n_machines=10)
    cfg = apply_shared_eval_args(get_default_eval_config(), args)
    assert cfg.env.n_machines == 10
    assert cfg.env.n_jobs == 15
    assert cfg.env.avg_operations_per_job == 8


def test_eval_config_from_args_full_scale():
    assert eval_config_from_args(None).env.n_machines == 5
    assert eval_config_from_args(Namespace(full_scale=False)).env.n_machines == 5
    assert eval_config_from_args(Namespace(full_scale=True)).env.n_machines == 25


def test_heuristic_and_milp_cli_full_scale_upgrades_demo_config():
    from baseline_heuristic import parse_args as parse_heuristic
    from baseline_milp import parse_args as parse_milp

    for parse, extra in (
        (parse_heuristic, ["--rule", "SPT"]),
        (parse_milp, []),
    ):
        args = parse([*extra, "--full-scale"])
        cfg = apply_shared_eval_args(get_default_eval_config(), args)
        assert cfg.env.n_machines == 25
        assert cfg.env.n_jobs == 15
        assert cfg.env.avg_operations_per_job == 8


def test_build_eval_train_config_uses_eval_seed():
    cfg = get_default_eval_config()
    cfg.seed = 11
    train_cfg = build_eval_train_config(cfg)
    assert train_cfg.n_envs == 1
    assert train_cfg.seed == 11
    assert train_cfg.eval_seed == 11
    assert train_cfg.env.n_machines == cfg.env.n_machines

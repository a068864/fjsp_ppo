from argparse import Namespace

import pytest

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


def test_heuristic_milp_and_lp_cli_full_scale_upgrades_demo_config():
    from baseline_heuristic import parse_args as parse_heuristic
    from baseline_lp import parse_args as parse_lp
    from baseline_milp import parse_args as parse_milp

    for parse, extra in (
        (parse_heuristic, ["--rule", "SPT"]),
        (parse_milp, []),
        (parse_lp, ["--rounding-trials", "4"]),
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


def test_compare_baselines_cli_flags():
    from compare_baselines import parse_args as parse_compare

    args = parse_compare(
        ["--trust-checkpoint", "--rounding-trials", "4", "--full-scale"]
    )
    assert args.trust_checkpoint is True
    assert args.rounding_trials == 4
    assert args.full_scale is True
    cfg = apply_shared_eval_args(get_default_eval_config(), args)
    assert cfg.env.n_machines == 25


def test_compare_table_reports_episode_latency_not_mixed_ms_step(capsys):
    """Constructive times are per step; MILP/LP times are per instance."""
    from compare_baselines import _episode_time_s, _print_table

    ppo = {
        "method": "PPO",
        "mean_makespan": 53.0,
        "std_makespan": 1.0,
        "success_rate": 1.0,
        "mean_ep_length": 12.0,
        "mean_inference_time_s": 0.006,
        "time_is_per_episode": False,
    }
    milp = {
        "method": "MILP",
        "mean_makespan": 50.0,
        "std_makespan": 1.0,
        "success_rate": 1.0,
        "mean_ep_length": 12.0,
        "mean_inference_time_s": 0.096,
        "time_is_per_episode": True,
    }
    assert _episode_time_s(ppo) == pytest.approx(0.072)
    assert _episode_time_s(milp) == pytest.approx(0.096)

    _print_table([milp, ppo])
    text = capsys.readouterr().out
    assert "ms/ep" in text
    assert "ms/step" not in text
    assert "LB_LP" not in text
    assert "Cmax/LB" not in text
    assert "72.000" in text
    assert "96.000" in text
    assert "  6.000" not in text


def test_compare_table_lists_lp_rules_even_if_worse(capsys):
    from compare_baselines import _lp_rule_row, _print_table
    from training.evaluate import EvalResult

    template = EvalResult(
        mean_makespan=50.0,
        std_makespan=1.0,
        mean_ep_length=12.0,
        std_ep_length=0.0,
        success_rate=1.0,
        mean_inference_time_s=0.02,
        n_episodes=2,
        n_success=2,
    )
    lrpt = _lp_rule_row("LRPT", [50.0, 52.0], template)
    cp = _lp_rule_row("CP", [60.0, 62.0], template)
    assert lrpt["method"] == "LP-LRPT"
    assert cp["method"] == "LP-CP"
    assert float(cp["mean_makespan"]) > float(lrpt["mean_makespan"])
    _print_table([lrpt, cp])
    text = capsys.readouterr().out
    assert "LP-LRPT" in text
    assert "LP-CP" in text

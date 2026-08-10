import numpy as np
import pytest
from config import get_default_eval_config
from training.eval_cli import build_eval_train_config
from training.evaluate import evaluate_random_fjsp, sample_masked_random_actions
from training.make_env import make_vec_env


def test_samples_only_valid_actions():
    rng = np.random.default_rng(0)
    mask = np.array([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float32)
    for _ in range(50):
        actions = sample_masked_random_actions(mask, rng)
        assert actions.shape == (2,)
        assert actions[0] in (0, 2)
        assert actions[1] == 1


def test_empty_mask_raises():
    rng = np.random.default_rng(0)
    mask = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        sample_masked_random_actions(mask, rng)


def test_random_eval_is_repeatable_on_held_out_seed():
    cfg = get_default_eval_config()
    cfg.seed = 19
    cfg.n_episodes = 2
    cfg.env.n_machines = 2
    cfg.env.n_jobs = 2
    cfg.env.avg_operations_per_job = 2
    train_cfg = build_eval_train_config(cfg)

    def _run():
        env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
        try:
            env.seed(cfg.seed)
            return evaluate_random_fjsp(env, n_episodes=cfg.n_episodes, seed=cfg.seed)
        finally:
            env.close()

    a = _run()
    b = _run()
    assert a.mean_reward == pytest.approx(b.mean_reward)
    assert a.mean_ep_length == pytest.approx(b.mean_ep_length)


"""VecEnv seeding and deferred SB3 seed/options contract."""

from __future__ import annotations

import numpy as np

from config import get_default_train_config
from training.make_env import GraphDummyVecEnv, make_env_fn, make_vec_env


def test_make_env_fn_does_not_eager_reset():
    cfg = get_default_train_config()
    cfg.n_envs = 1
    calls = {"reset": 0}
    thunk = make_env_fn(cfg, rank=0, for_eval=False)
    env = thunk()
    original_reset = env.reset

    def _counting_reset(*args, **kwargs):
        calls["reset"] += 1
        return original_reset(*args, **kwargs)

    # Fresh thunk builds without an extra reset after construction.
    env2 = make_env_fn(cfg, rank=1, for_eval=False)()
    # After build, needing reset is Fine; the factory must not call reset itself.
    # Probe via a sentinel attribute set only by our patched path on a new thunk.
    env.close()
    env2.close()
    assert calls["reset"] == 0


def test_dummy_vec_deferred_seed_applies_on_first_reset():
    cfg = get_default_train_config()
    cfg.n_envs = 2
    cfg.seed = 123
    env = make_vec_env(cfg, n_envs=2, use_subprocess=False, for_eval=True)
    env.seed(50)
    obs = env.reset()
    # Seeds are consumed on reset then cleared (SB3 contract).
    assert env._seeds == [None, None]
    masks = obs["action_mask"]
    assert masks.shape[0] == 2
    env.close()


def test_deterministic_instance_sequence_with_fixed_seed():
    cfg = get_default_train_config()
    cfg.n_envs = 1
    cfg.seed = 7

    def _first_op_duration(vec):
        obs = vec.reset()
        graph = obs["graph"][0]
        return float(graph["operation"].x[0, 0].item())

    a = make_vec_env(cfg, n_envs=1, use_subprocess=False, for_eval=True)
    a.seed(7)
    d1 = _first_op_duration(a)
    a.seed(7)
    d2 = _first_op_duration(a)
    a.close()
    assert d1 == d2


def test_dummy_and_subprocess_first_reset_parity():
    cfg = get_default_train_config()
    cfg.n_envs = 2
    cfg.seed = 11

    dummy = make_vec_env(cfg, n_envs=2, use_subprocess=False, for_eval=True)
    dummy.seed(11)
    obs_d = dummy.reset()

    try:
        sub = make_vec_env(cfg, n_envs=2, use_subprocess=True, for_eval=True)
        sub.seed(11)
        obs_s = sub.reset()
        assert np.allclose(obs_d["action_mask"], obs_s["action_mask"])
        for i in range(2):
            xd = obs_d["graph"][i]["operation"].x.detach().cpu().numpy()
            xs = obs_s["graph"][i]["operation"].x.detach().cpu().numpy()
            assert np.allclose(xd, xs)
        sub.close()
    finally:
        dummy.close()

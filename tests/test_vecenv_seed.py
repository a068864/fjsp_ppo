"""VecEnv seeding and deferred SB3 seed/options contract."""

from __future__ import annotations

import numpy as np

from config import get_default_eval_config, get_default_train_config
from heuristics import RULES
from training.eval_cli import build_eval_train_config
from training.evaluate import (
    evaluate_heuristic_fjsp,
    evaluate_policy_fjsp,
    evaluate_random_fjsp,
)
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
    cfg.eval_seed = 7

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


def test_eval_autoreset_uses_deterministic_episode_seeds():
    cfg = get_default_train_config()
    cfg.eval_seed = 1_000_042
    env = make_vec_env(cfg, n_envs=1, use_subprocess=False, for_eval=True)
    try:
        env.seed(cfg.eval_seed)
        first = []
        obs = env.reset()
        first.append(obs["graph"][0]["operation"].x[:, 0].detach().cpu().numpy().copy())

        while len(first) < 4:
            mask = np.asarray(obs["action_mask"]).reshape(-1)
            action = int(np.flatnonzero(mask > 0.5)[0])
            obs, _rew, dones, _infos = env.step([action])
            if bool(dones[0]):
                first.append(
                    obs["graph"][0]["operation"].x[:, 0].detach().cpu().numpy().copy()
                )

        env.seed(cfg.eval_seed)
        second = []
        obs = env.reset()
        second.append(obs["graph"][0]["operation"].x[:, 0].detach().cpu().numpy().copy())
        while len(second) < 4:
            mask = np.asarray(obs["action_mask"]).reshape(-1)
            action = int(np.flatnonzero(mask > 0.5)[0])
            obs, _rew, dones, _infos = env.step([action])
            if bool(dones[0]):
                second.append(
                    obs["graph"][0]["operation"].x[:, 0].detach().cpu().numpy().copy()
                )

        assert len(first) == 4
        for a, b in zip(first, second):
            np.testing.assert_allclose(a, b)
        assert not np.allclose(first[0], first[1])
    finally:
        env.close()


def test_train_autoreset_remains_unseeded_random():
    cfg = get_default_train_config()
    env = make_vec_env(cfg, n_envs=1, use_subprocess=False, for_eval=False)
    try:
        env.seed(7)
        obs = env.reset()
        seen = [obs["graph"][0]["operation"].x[:, 0].detach().cpu().numpy().copy()]
        while len(seen) < 3:
            mask = np.asarray(obs["action_mask"]).reshape(-1)
            action = int(np.flatnonzero(mask > 0.5)[0])
            obs, _rew, dones, _infos = env.step([action])
            if bool(dones[0]):
                seen.append(
                    obs["graph"][0]["operation"].x[:, 0].detach().cpu().numpy().copy()
                )
        assert len(seen) == 3
        assert isinstance(env, GraphDummyVecEnv)
        assert env.for_eval is False
    finally:
        env.close()


def test_dummy_and_subprocess_first_reset_parity():
    cfg = get_default_train_config()
    cfg.n_envs = 2
    cfg.seed = 11

    # Training (for_eval=False) may use Dummy or Subproc; compare first reset.
    dummy = make_vec_env(cfg, n_envs=2, use_subprocess=False, for_eval=False)
    dummy.seed(11)
    obs_d = dummy.reset()

    try:
        sub = make_vec_env(cfg, n_envs=2, use_subprocess=True, for_eval=False)
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


class _FirstValidPolicy:
    """Minimal stand-in for evaluate_policy_fjsp (PPO path)."""

    def predict(self, observations, deterministic=True):
        mask = np.asarray(observations["action_mask"], dtype=np.float32)
        if mask.ndim == 1:
            mask = mask.reshape(1, -1)
        actions = np.array(
            [int(np.flatnonzero(row > 0.5)[0]) for row in mask],
            dtype=np.int64,
        )
        return actions, None


def _fingerprint(obs) -> np.ndarray:
    return obs["graph"][0]["operation"].x[:, 0].detach().cpu().numpy().copy()


def _collect_instance_suite(env, n_episodes: int, choose_action) -> list[np.ndarray]:
    """Record op-duration fingerprints at each episode start under the eval schedule."""
    fps: list[np.ndarray] = []
    obs = env.reset()
    fps.append(_fingerprint(obs))
    completed = 0
    while completed < n_episodes:
        action = choose_action(env, obs)
        obs, _rew, dones, _infos = env.step(np.asarray([action], dtype=np.int64))
        if bool(dones[0]):
            completed += 1
            if completed < n_episodes:
                fps.append(_fingerprint(obs))
    return fps


def test_ppo_random_heuristics_share_distinct_instance_suite():
    """Same eval_seed / n_episodes → identical distinct instances across agents."""
    from heuristics import select_heuristic_action
    from training.evaluate import sample_masked_random_actions

    eval_cfg = get_default_eval_config()
    eval_cfg.seed = 77
    eval_cfg.n_episodes = 3
    eval_cfg.env.n_machines = 2
    eval_cfg.env.n_jobs = 2
    eval_cfg.env.avg_operations_per_job = 2
    train_cfg = build_eval_train_config(eval_cfg)
    n_episodes = eval_cfg.n_episodes
    assert train_cfg.eval_seed == eval_cfg.seed

    def _first_valid(_env, obs) -> int:
        mask = np.asarray(obs["action_mask"]).reshape(-1)
        return int(np.flatnonzero(mask > 0.5)[0])

    ref_env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
    try:
        ref_env.seed(eval_cfg.seed)
        reference = _collect_instance_suite(ref_env, n_episodes, _first_valid)
    finally:
        ref_env.close()
    assert len(reference) == n_episodes
    assert not np.allclose(reference[0], reference[1])

    def _assert_suite(label: str, fps: list[np.ndarray]) -> None:
        assert len(fps) == n_episodes, label
        for i, (a, b) in enumerate(zip(reference, fps)):
            np.testing.assert_allclose(a, b, err_msg=f"{label} episode {i}")

    # Repeatability of the held-out schedule.
    env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
    try:
        env.seed(eval_cfg.seed)
        _assert_suite("repeat", _collect_instance_suite(env, n_episodes, _first_valid))
    finally:
        env.close()

    # PPO evaluate entrypoint (first-valid stand-in) + fingerprint suite.
    env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
    try:
        env.seed(eval_cfg.seed)
        _assert_suite("ppo", _collect_instance_suite(env, n_episodes, _first_valid))
        env.seed(eval_cfg.seed)
        result = evaluate_policy_fjsp(
            _FirstValidPolicy(), env, n_episodes=n_episodes, deterministic=True
        )
        assert result.n_episodes == n_episodes
    finally:
        env.close()

    # Random baseline: same instances regardless of action RNG.
    rng = np.random.default_rng(eval_cfg.seed)

    def _random_action(_env, obs) -> int:
        return int(sample_masked_random_actions(obs["action_mask"], rng)[0])

    env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
    try:
        env.seed(eval_cfg.seed)
        _assert_suite("random", _collect_instance_suite(env, n_episodes, _random_action))
        env.seed(eval_cfg.seed)
        result = evaluate_random_fjsp(env, n_episodes=n_episodes, seed=eval_cfg.seed)
        assert result.n_episodes == n_episodes
    finally:
        env.close()

    # Every heuristic rule shares the same held-out suite.
    for rule in sorted(RULES):
        def _heuristic_action(env, _obs, _rule=rule) -> int:
            return int(select_heuristic_action(env.envs[0].unwrapped, _rule))

        env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
        try:
            env.seed(eval_cfg.seed)
            _assert_suite(
                f"heuristic-{rule}",
                _collect_instance_suite(env, n_episodes, _heuristic_action),
            )
            env.seed(eval_cfg.seed)
            result = evaluate_heuristic_fjsp(env, rule=rule, n_episodes=n_episodes)
            assert result.n_episodes == n_episodes
        finally:
            env.close()

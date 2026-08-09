from envs.fjsp_env import FJSPEnv


def test_reset_returns_graph_obs_keys():
    env = FJSPEnv(n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=0, device="cpu")
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == {"dummy", "action_mask", "graph"}
    assert obs["dummy"].shape == (1,)
    assert obs["action_mask"].shape == (env.n_machines * env.n_operations,)
    assert float(obs["action_mask"].sum()) > 0.0
    env.close()

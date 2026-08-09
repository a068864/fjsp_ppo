from gymnasium import spaces
from envs.fjsp_env import GraphObsSpace, make_sb3_graph_observation_space
from training.make_env import stack_fjsp_obs
from torch_geometric.data import HeteroData
import numpy as np


def test_sb3_space_is_dict_with_expected_keys():
    space = make_sb3_graph_observation_space(10)
    assert isinstance(space, spaces.Dict)
    assert set(space.spaces.keys()) == {"dummy", "action_mask", "graph"}
    assert space.spaces["action_mask"].shape == (10,)


def test_graph_obs_space_contains_valid_obs():
    gspace = GraphObsSpace(6)
    obs = {
        "dummy": np.zeros((1,), dtype=np.float32),
        "action_mask": np.ones((6,), dtype=np.float32),
        "graph": HeteroData(),
    }
    assert gspace.contains(obs)


def test_stack_preserves_heterodata_objects():
    g1, g2 = HeteroData(), HeteroData()
    stacked = stack_fjsp_obs(
        [
            {
                "dummy": np.zeros((1,), dtype=np.float32),
                "action_mask": np.ones((4,), dtype=np.float32),
                "graph": g1,
            },
            {
                "dummy": np.zeros((1,), dtype=np.float32),
                "action_mask": np.ones((4,), dtype=np.float32),
                "graph": g2,
            },
        ]
    )
    assert stacked["dummy"].shape == (2, 1)
    assert stacked["action_mask"].shape == (2, 4)
    assert stacked["graph"].dtype == object
    assert stacked["graph"][0] is g1

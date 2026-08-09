import pytest
from utils import unflatten_action


def test_unflatten_action_roundtrip_indices():
    assert unflatten_action(0, 8) == (0, 0)
    assert unflatten_action(7, 8) == (0, 7)
    assert unflatten_action(8, 8) == (1, 0)


def test_unflatten_rejects_bad_n_operations():
    with pytest.raises(ValueError):
        unflatten_action(0, 0)

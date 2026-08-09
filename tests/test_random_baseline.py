import numpy as np
import pytest
from training.evaluate import sample_masked_random_actions


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

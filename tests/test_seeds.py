"""Tests for the global seed control in pvforecast.seeds."""

import numpy as np
import pytest

from pvforecast import seeds


def test_set_seed_makes_numpy_reproducible():
    seeds.set_seed(7)
    first = np.random.rand(5)

    seeds.set_seed(7)
    second = np.random.rand(5)

    np.testing.assert_array_equal(first, second)


def test_set_seed_returns_the_seed():
    assert seeds.set_seed(11) == 11


def test_set_seed_rejects_a_negative_seed():
    with pytest.raises(ValueError, match="Seed muss"):
        seeds.set_seed(-1)


def test_model_seeds_allow_a_spread():
    # Single-seed results hide the run-to-run variation of the stochastic models.
    assert len(set(seeds.MODEL_SEEDS)) > 1

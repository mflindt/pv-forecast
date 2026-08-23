"""
Global seed control.

Every run sets the seed once and records it in the artefacts, so a rerun reproduces
the same searches and fits.
"""

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42

# Repeats for the stochastic models.
MODEL_SEEDS = (42, 43, 44)


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed the interpreter, NumPy and the hash randomisation."""
    if seed < 0:
        raise ValueError(f"Seed muss >= 0 sein, ist {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Seed gesetzt: {seed}")
    return seed

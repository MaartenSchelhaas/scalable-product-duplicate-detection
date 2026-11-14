"""
Derive one seed per bootstrap from the master seed; every random step of a bootstrap uses that seed
"""

import numpy as np


def make_seeds(master_seed: int, n_bootstraps: int) -> list[int]:
    """Derive one seed per bootstrap from the master seed.

    Args:
        master_seed (int): The master seed from config
        n_bootstraps (int): Number of bootstraps

    Returns:
        list[int]: One seed per bootstrap
    """
    return np.random.SeedSequence(master_seed).generate_state(n_bootstraps).tolist()
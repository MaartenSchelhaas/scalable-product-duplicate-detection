"""
Hash embedding vectors into n-bit signatures with random hyperplanes, the input of the LSH in lsh.py.
"""

import numpy as np


def signatures(vectors: np.ndarray, n_bits: int, seed: int) -> np.ndarray:
    """Random-hyperplane signature of every product, computed once per model and bootstrap.

    Args:
        vectors (np.ndarray): Embedding of every product, row i belongs to products[i]
        n_bits (int): Number of random hyperplanes, the length of each signature
        seed (int): Seed of the hyperplanes; the same seed gives the same hyperplanes for every model

    Returns:
        np.ndarray: Boolean signature matrix (n_bits, number of products), column i is
        the signature of products[i], row k is True where a product lies on the positive side of hyperplane k
    """
    rng = np.random.default_rng(seed)
    dimension = vectors.shape[1]

    # Row k is the normal vector of hyperplane k, drawn from a standard normal distribution in every direction
    hyperplanes = rng.standard_normal((n_bits, dimension))

    # projection of the embedding vectors on the hyperplanes
    projections = hyperplanes @ vectors.T
    signature_matrix = projections > 0
    return signature_matrix

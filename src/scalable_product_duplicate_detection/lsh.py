"""
File to do the locality sensitive hashing. Such that we only
compare similar products instead of all.
LSH: Split each signature vector into b bands of r rows.
Products whose rows are the same in at least one band share
a bucket and become a candidate pair. More similar products
have a higher chance to be in the same bucket.

Every similarity method makes its own signature matrix (e.g. random hyperplanes for embeddings); the banding is shared.
"""

import itertools

import numpy as np


def candidate_pairs(signatures: np.ndarray, subset_indices: list[int], rows: int) -> list[tuple[int, int]]:
    """Pairs of products that share a bucket in at least one band.

    The rows of the signature matrix are cut into b = n_bits // rows bands of rows rows; leftover rows are unused.
    Signatures holds every product, but only the products in subset_indices go into the buckets.

    Args:
        signatures (np.ndarray): Signature matrix of all products, column i belongs to products[i]
        subset_indices (list[int]): Positions in products of the products to hash, e.g. train or test split
        rows (int): Rows per band r; a larger r gives fewer candidates

    Returns:
        list[tuple[int, int]]: Every candidate pair once, as (smaller index, larger index), sorted
    """
    n_bits = signatures.shape[0]
    if rows < 1 or rows > n_bits:
        raise ValueError(f"rows must lie between 1 and n_bits ({n_bits}), got {rows}")

    bands = n_bits // rows

    # A set, so a pair that shares a bucket in several bands is kept once
    pairs: set[tuple[int, int]] = set()

    for band in range(bands):
        start = band * rows
        buckets = _buckets_of_band(signatures, subset_indices, start, rows)
        pairs_of_band = _pairs_in_buckets(buckets)

        for pair in pairs_of_band:
            pairs.add(pair)

    sorted_pairs = sorted(pairs)
    return sorted_pairs


def _buckets_of_band(
    signatures: np.ndarray,
    subset_indices: list[int],
    start: int,
    rows: int,
) -> dict[bytes, list[int]]:
    """Products grouped by their rows start to start + rows: equal bits give the same key, hence the same bucket."""
    buckets: dict[bytes, list[int]] = {}

    for index in subset_indices:
        band_bits = signatures[start : start + rows, index]
        key = band_bits.tobytes()

        if key not in buckets:
            buckets[key] = []

        buckets[key].append(index)

    return buckets


def _pairs_in_buckets(buckets: dict[bytes, list[int]]) -> set[tuple[int, int]]:
    """Every pair of products that share a bucket, as (smaller index, larger index)."""
    pairs: set[tuple[int, int]] = set()

    for members in buckets.values():
        for i, j in itertools.combinations(sorted(members), 2):
            pairs.add((i, j))

    return pairs

"""
Single-linkage clustering with the merge rule of MSM: two clusters merge only if every pair across them is finite.
A pair is infinite if it is not a candidate, if both products come from the same shop, or if their known brands differ.
"""

from dataclasses import dataclass

import numpy as np

from scalable_product_duplicate_detection.data import PreparedProduct
from scalable_product_duplicate_detection.evaluation import cluster_scores


@dataclass(frozen=True)
class _Merge:
    distance: float
    first: int
    second: int


def cluster(
    vectors: np.ndarray,
    products: list[PreparedProduct],
    subset_indices: list[int],
    candidate_pairs: list[tuple[int, int]],
    epsilons: list[float],
) -> dict[float, list[list[int]]]:
    """Cluster a set of products once for every epsilon.

    The main part (distances, sorting, single linkage) runs once. After which for each epsilon
    we can quickly get the clusters. 

    Args:
        vectors (np.ndarray): Unit-length embedding of every product, row i belongs to products[i]
        products (list[PreparedProduct]): All products, in load_products order (shop and brand decide the infinite pairs)
        subset_indices (list[int]): Positions in products(list) of the products to cluster
        candidate_pairs (list[tuple[int, int]]): Pairs of product indices to compare. Non pairs will receive inf distance.
        epsilons (list[float]): Largest distance at which clusters still merge, one clustering per value

    Returns:
        dict[float, list[list[int]]]: The clusters for each epsilon: For each epsilon, for each cluster a list of indices
    """
    distances = _candidate_distances(vectors, products, candidate_pairs)
    merges = _single_linkage(distances)

    clusterings = {}
    for epsilon in epsilons:
        clusterings[epsilon] = _clusters_for_epsilon(merges, subset_indices, epsilon)

    return clusterings


def best_epsilon(
    clusterings: dict[float, list[list[int]]],
    products: list[PreparedProduct],
    subset_indices: list[int],
) -> float:
    """The epsilon whose clustering has the highest F1.

    Counts F1 with the modelIDs of the clustered products.

    Args:
        clusterings (dict[float, list[list[int]]]): Output of cluster() over the epsilon grid
        products (list[PreparedProduct]): All products, in load_products order
        subset_indices (list[int]): The subset_indices given to cluster()

    Returns:
        float: The best epsilon; on a tie the smallest
    """
    best = None
    # Below every possible F1, so the first epsilon always wins the first comparison, also when all F1 are 0
    best_f1 = -1.0

    for epsilon, clusters in clusterings.items():
        f1 = cluster_scores(clusters, products, subset_indices)["f1"]

        # Strictly greater, on a tie the smaller epsilon stays
        if f1 > best_f1:
            best = epsilon
            best_f1 = f1

    if best is None:
        raise ValueError("clusterings is empty")

    return best


def _candidate_distances(
    vectors: np.ndarray,
    products: list[PreparedProduct],
    candidate_pairs: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Distance 1 - cosine per finite candidate pair, keyed (smaller index, larger index); infinite pairs are left out."""
    distances = {}

    for pair in candidate_pairs:
        # _can_merge looks pairs up with the smaller index first, so every key is stored that way
        i, j = min(pair), max(pair)

        # Pairs with infinite distance are dropped as candidate pairs
        if _is_infinite(products[i], products[j]):
            continue

        # The vectors have length 1, so their dot product is the cosine similarity; a pair given twice gets the same key
        cosine = float(vectors[i] @ vectors[j])
        distances[(i, j)] = 1.0 - cosine

    return distances


def _single_linkage(distances: dict[tuple[int, int], float]) -> list[_Merge]:
    """Full merge history of single linkage with the MSM rule, without a threshold.

    Walks all finite pairs from small to large distance and joins the clusters of a pair if they are not together yet
    and every pair across them is finite; a refused merge stays refused, as clusters only grow. Epsilon is applied
    afterwards: _clusters_for_epsilon replays the history up to the first merge above epsilon.
    """
    #Products which are alone are not in cluster_of_product
    cluster_of_product: dict[int, list[int]] = {}
    merges = []

    for (i, j), distance in sorted(distances.items(), key=lambda item: item[1]):
        first = cluster_of_product.get(i, [i])
        second = cluster_of_product.get(j, [j])

        if j in first:
            continue

        if not _can_merge(first, second, distances):
            continue

        merged = first + second
        for member in merged:
            cluster_of_product[member] = merged

        merges.append(_Merge(distance=distance, first=i, second=j))

    return merges


def _clusters_for_epsilon(merges: list[_Merge], subset_indices: list[int], epsilon: float) -> list[list[int]]:
    """Clusters after every merge with a distance of at most epsilon, singletons included."""
    cluster_of_product = {index: [index] for index in subset_indices}

    for merge in merges:
        # The merges are sorted by distance, so the first one above epsilon ends the replay
        if merge.distance > epsilon:
            break

        # Same state as when _single_linkage accepted this merge, so it is valid again
        merged = cluster_of_product[merge.first] + cluster_of_product[merge.second]
        for member in merged:
            cluster_of_product[member] = merged

    # Every member of a cluster points to the same list, so each cluster appears once per member; the set keeps it once
    unique_clusters = set()
    for members in cluster_of_product.values():
        unique_clusters.add(tuple(sorted(members)))

    # Sorted, so the same clustering always comes out in the same order
    clusters = []
    for members in sorted(unique_clusters):
        clusters.append(list(members))

    return clusters


def _is_infinite(first: PreparedProduct, second: PreparedProduct) -> bool:
    """Same shop, or two known brands that differ. Return False if pair
    should not be dropped."""
    if first.shop == second.shop:
        return True

    # An unknown brand never rules a pair out
    if first.brand is not None and second.brand is not None and first.brand != second.brand:
        return True

    return False


def _can_merge(first: list[int], second: list[int], distances: dict[tuple[int, int], float]) -> bool:
    """MSM rule: every pair across the two clusters must be finite, i.e. present in distances."""
    for i in first:
        for j in second:
            #Cross pairs are per definition unordered
            if (min(i, j), max(i, j)) not in distances:
                return False

    return True
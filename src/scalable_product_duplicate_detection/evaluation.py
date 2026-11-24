"""
File to get evaluation metrics, like:
pair counting, quality, completeness, F1 ...

Every metric counts pairs of products, each pair stored as (smaller index, larger index).
"""

import itertools
import math

from scalable_product_duplicate_detection.data import PreparedProduct


def cluster_scores(
    clusters: list[list[int]],
    products: list[PreparedProduct],
    subset_indices: list[int],
) -> dict[str, float]:
    """Precision, recall and F1 of a clustering, counted on pairs.

    Args:
        clusters (list[list[int]]): Predicted clusters of product indices, e.g. one clustering from cluster()
        products (list[PreparedProduct]): All products, in load_products order (modelID gives the real duplicates)
        subset_indices (list[int]): Positions in products of the clustered products, the same set given to cluster()

    Returns:
        dict[str, float]: tp, fp, fn, precision, recall and f1; a ratio with an empty denominator is 0.0
    """
    predicted = _pairs_in_clusters(clusters)
    duplicates = _pairs_in_clusters(_real_clusters(products, subset_indices))
    scores = _pair_counts(predicted, duplicates)
    return scores


def lsh_scores(
    candidate_pairs: list[tuple[int, int]],
    products: list[PreparedProduct],
    subset_indices: list[int],
) -> dict[str, float]:
    """Pair quality, pair completeness, F1* and fraction of comparisons of the LSH candidates.

    Every candidate pair counts as a comparison; the fraction divides them by all pairs among subset_indices.

    Args:
        candidate_pairs (list[tuple[int, int]]): Output of candidate_pairs() in lsh.py
        products (list[PreparedProduct]): All products, in load_products order (modelID gives the real duplicates)
        subset_indices (list[int]): Positions in products of the hashed products, the same set given to candidate_pairs()

    Returns:
        dict[str, float]: comparisons, possible_comparisons, fraction_of_comparisons, tp, fp, fn, pair_quality,
        pair_completeness and f1_star
    """
    candidates = set(candidate_pairs)
    duplicates = _pairs_in_clusters(_real_clusters(products, subset_indices))

    comparisons = len(candidates)
    possible_comparisons = math.comb(len(subset_indices), 2)
    fraction_of_comparisons = _ratio(comparisons, possible_comparisons)

    # Precision, recall and F1 of the candidate pairs are what the assignment calls PQ, PC and F1*
    counts = _pair_counts(candidates, duplicates)
    pair_quality = counts["precision"]
    pair_completeness = counts["recall"]
    f1_star = counts["f1"]

    scores = {
        "comparisons": comparisons,
        "possible_comparisons": possible_comparisons,
        "fraction_of_comparisons": fraction_of_comparisons,
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "pair_quality": pair_quality,
        "pair_completeness": pair_completeness,
        "f1_star": f1_star,
    }
    return scores


def _pairs_in_clusters(clusters: list[list[int]]) -> set[tuple[int, int]]:
    """Every pair of products that share a cluster."""
    pairs = set()

    for members in clusters:
        # Sorted, so every pair comes out as (smaller index, larger index) and matches the same pair from the other side
        for i, j in itertools.combinations(sorted(members), 2):
            pairs.add((i, j))

    return pairs


def _real_clusters(products: list[PreparedProduct], subset_indices: list[int]) -> list[list[int]]:
    """The clustered products grouped by modelID, same-shop duplicates included (those are always FN)."""
    members_of_model: dict[str, list[int]] = {}

    for index in subset_indices:
        model_id = products[index].model_id

        # The first product of a modelID starts its group
        if model_id not in members_of_model:
            members_of_model[model_id] = []

        members_of_model[model_id].append(index)

    clusters = list(members_of_model.values())
    return clusters


def _pair_counts(predicted: set[tuple[int, int]], duplicates: set[tuple[int, int]]) -> dict[str, float]:
    """TP, FP and FN of the predicted pairs, with the precision, recall and harmonic mean that follow from them."""
    tp = len(predicted & duplicates)
    fp = len(predicted) - tp
    fn = len(duplicates) - tp

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)

    scores = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    return scores


def _ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, or 0.0 if the denominator is 0 (no predicted pairs, or no duplicates)."""
    if denominator == 0:
        return 0.0

    return numerator / denominator
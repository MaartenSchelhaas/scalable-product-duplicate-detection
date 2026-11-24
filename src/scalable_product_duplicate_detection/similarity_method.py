"""
Similarity method: how a method represents, hashes and compares products, with the steps every method shares.

A subclass implements what makes it that method:
1) fit: learn from the training part (or load what was learned before), represent and hash every product
2) distance: the distance between two products
3) parameter_grid: the settings it tunes besides epsilon
The base class does the rest the same way for every method: LSH candidates, tuning on the tuning part and
finding the duplicates in the test part.
"""

from abc import ABC, abstractmethod

import numpy as np

from scalable_product_duplicate_detection import lsh
from scalable_product_duplicate_detection.bootstrap import Split
from scalable_product_duplicate_detection.clustering import best_epsilon, cluster
from scalable_product_duplicate_detection.data import PreparedProduct


class SimilarityMethod(ABC):
    """One duplicate detection method for one bootstrap.

    Use: fit(split) once, then per r and linkage: tune(r, linkage) on the tuning part and
    find_duplicates(test, r, linkage) in the test part.
    """

    # Set by fit: column i is the signature of products[i], and the part the method tunes on
    signature_matrix: np.ndarray
    tuning_indices: list[int]

    def __init__(self, products: list[PreparedProduct], epsilons: list[float]) -> None:
        """
        Args:
            products (list[PreparedProduct]): All products, in load_products order
            epsilons (list[float]): Epsilon grid that tune searches
        """
        self.products = products
        self.epsilons = epsilons

        # The setting distance uses; tune and find_duplicates set it from the grid
        self.parameters: dict = {}

        # Per (r, linkage) the best setting and epsilon from tune, read by find_duplicates
        self.tuned: dict[tuple[int, str], tuple[dict, float]] = {}

    @abstractmethod
    def fit(self, split: Split) -> None:
        """Learn from the training part of split (or load it), then represent and hash every product.

        Must set signature_matrix and tuning_indices.
        """

    @abstractmethod
    def distance(self, i: int, j: int) -> float:
        """Distance between products[i] and products[j] under the current parameters."""

    @abstractmethod
    def parameter_grid(self) -> list[dict]:
        """Every setting tune tries besides epsilon; [{}] if the method only tunes epsilon."""

    def candidate_pairs(self, subset_indices: list[int], rows_per_band: int) -> list[tuple[int, int]]:
        """LSH candidate pairs among subset_indices, with r = rows_per_band."""
        return lsh.candidate_pairs(self.signature_matrix, subset_indices, rows_per_band)

    def tune(self, rows_per_band: int, linkage: str) -> None:
        """The setting and epsilon with the best F1 on the tuning candidates, stored in tuned[(rows_per_band, linkage)].

        On a tie the first setting of the grid and the smallest epsilon stay.

        Args:
            rows_per_band (int): r of the LSH
            linkage (str): One of clustering.LINKAGES
        """
        candidates = self.candidate_pairs(self.tuning_indices, rows_per_band)

        best = None
        # Below every possible F1, so the first setting always wins the first comparison
        best_f1 = -1.0

        for parameters in self.parameter_grid():
            self.parameters = parameters
            clusterings = cluster(
                self.distance, self.products, self.tuning_indices, candidates, self.epsilons, linkage
            )
            epsilon, f1 = best_epsilon(clusterings, self.products, self.tuning_indices)

            if f1 > best_f1:
                best = (parameters, epsilon)
                best_f1 = f1

        if best is None:
            raise ValueError("parameter_grid is empty")

        self.tuned[(rows_per_band, linkage)] = best

    def find_duplicates(
        self,
        subset_indices: list[int],
        rows_per_band: int,
        linkage: str,
    ) -> tuple[list[tuple[int, int]], list[list[int]]]:
        """Candidate pairs and clusters of subset_indices with the setting and epsilon that tune stored.

        Args:
            subset_indices (list[int]): Positions in products of the products to search, e.g. the test part
            rows_per_band (int): r of the LSH, tuned before with the same linkage
            linkage (str): One of clustering.LINKAGES

        Returns:
            tuple[list[tuple[int, int]], list[list[int]]]: The LSH candidate pairs and the clusters
        """
        parameters, epsilon = self.tuned[(rows_per_band, linkage)]
        self.parameters = parameters

        candidates = self.candidate_pairs(subset_indices, rows_per_band)
        clusters = cluster(self.distance, self.products, subset_indices, candidates, [epsilon], linkage)[epsilon]
        return candidates, clusters

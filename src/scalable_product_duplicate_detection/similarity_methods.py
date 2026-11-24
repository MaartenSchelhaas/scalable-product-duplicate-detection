"""
Similarity methods: how a method represents, hashes and compares products, with the steps every method shares.

A subclass implements what makes it that method:
1) fit: learn from the training part (or load what was learned before), represent and hash every product
2) distance: the distance between two products
3) parameter_grid: the settings it tunes besides epsilon
The base class does the rest the same way for every method: LSH candidates, tuning on the tuning part and
finding the duplicates in the test part.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from scalable_product_duplicate_detection import lsh
from scalable_product_duplicate_detection.bootstrap import Split
from scalable_product_duplicate_detection.clustering import best_epsilon, cluster
from scalable_product_duplicate_detection.data import PreparedProduct, serialize
from scalable_product_duplicate_detection.embedding import embed, load_model
from scalable_product_duplicate_detection.finetune import finetune


class SimilarityMethod(ABC):
    """One duplicate detection method for one bootstrap.

    Use: fit(split) once, then per r: tune(r) on the tuning part and find_duplicates(test, r) in the test part.
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

        # Per r the best setting and epsilon from tune, read by find_duplicates
        self.tuned: dict[int, tuple[dict, float]] = {}

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

    def tune(self, rows_per_band: int) -> None:
        """The setting and epsilon with the best F1 on the tuning candidates, stored in tuned[rows_per_band].

        On a tie the first setting of the grid and the smallest epsilon stay.
        """
        candidates = self.candidate_pairs(self.tuning_indices, rows_per_band)

        best = None
        # Below every possible F1, so the first setting always wins the first comparison
        best_f1 = -1.0

        for parameters in self.parameter_grid():
            self.parameters = parameters
            clusterings = cluster(self.distance, self.products, self.tuning_indices, candidates, self.epsilons)
            epsilon, f1 = best_epsilon(clusterings, self.products, self.tuning_indices)

            if f1 > best_f1:
                best = (parameters, epsilon)
                best_f1 = f1

        if best is None:
            raise ValueError("parameter_grid is empty")

        self.tuned[rows_per_band] = best

    def find_duplicates(
        self,
        subset_indices: list[int],
        rows_per_band: int,
    ) -> tuple[list[tuple[int, int]], list[list[int]]]:
        """Candidate pairs and clusters of subset_indices with the setting and epsilon that tune stored for this r.

        Returns:
            tuple[list[tuple[int, int]], list[list[int]]]: The LSH candidate pairs and the clusters
        """
        parameters, epsilon = self.tuned[rows_per_band]
        self.parameters = parameters

        candidates = self.candidate_pairs(subset_indices, rows_per_band)
        clusters = cluster(self.distance, self.products, subset_indices, candidates, [epsilon])[epsilon]
        return candidates, clusters


class EmbeddingMethod(SimilarityMethod):
    """Embedding model with random-hyperplane LSH and distance 1 - cosine.

    A fine-tuned run loads its model from model_dir; if that folder does not exist yet, or retrain is True, fit first
    fine-tunes a fresh copy of the base model on the fine-tune part and saves it there.
    """

    def __init__(
        self,
        products: list[PreparedProduct],
        epsilons: list[float],
        base_model_dir: Path,
        model_dir: Path,
        fine_tuned: bool,
        max_tokens: int,
        n_bits: int,
        seed: int,
        epochs: int,
        groups_per_batch: int,
        batch_size: int,
        learning_rate: float,
        warmup_fraction: float,
        temperature: float,
        retrain: bool = False,
    ) -> None:
        """
        Args:
            products (list[PreparedProduct]): All products, in load_products order
            epsilons (list[float]): Epsilon grid that tune searches
            base_model_dir (Path): Folder of the base model, the starting point of fine-tuning
            model_dir (Path): Folder of the model: the base model, or the fine-tuned model of this bootstrap
            fine_tuned (bool): Whether model_dir holds (or gets) a model fine-tuned on the fine-tune part of this bootstrap
            max_tokens (int): Number of tokens after which a serialized product is truncated
            n_bits (int): Number of random hyperplanes, the length of each signature
            seed (int): Seed of this bootstrap, for the hyperplanes and the fine-tuning
            epochs, groups_per_batch, batch_size, learning_rate, warmup_fraction, temperature: Settings of finetune()
                in finetune.py, only used when fit fine-tunes
            retrain (bool): Fine-tune again even if model_dir exists
        """
        super().__init__(products, epsilons)
        self.base_model_dir = base_model_dir
        self.model_dir = model_dir
        self.fine_tuned = fine_tuned
        self.max_tokens = max_tokens
        self.n_bits = n_bits
        self.seed = seed
        self.epochs = epochs
        self.groups_per_batch = groups_per_batch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction
        self.temperature = temperature
        self.retrain = retrain

        # Epochs, losses and seconds of a fine-tuning run in fit; None if the model was loaded
        self.finetune_log: dict | None = None

    def fit(self, split: Split) -> None:
        """Load (or first fine-tune) the model, embed every product and hash the embeddings."""
        texts = []
        for product in self.products:
            texts.append(serialize(product))

        if self.fine_tuned and (self.retrain or not self.model_dir.exists()):
            model = self._finetune(split, texts)
        else:
            model = load_model(self.model_dir, self.max_tokens)

        self.vectors = embed(model, texts)
        self.signature_matrix = lsh.signatures(self.vectors, self.n_bits, self.seed)

        # A fine-tuned model has seen the fine-tune part, so it tunes epsilon on the validation part only
        if self.fine_tuned:
            self.tuning_indices = split.validation
        else:
            self.tuning_indices = split.train

    def distance(self, i: int, j: int) -> float:
        """1 - cosine; the vectors have length 1, so their dot product is the cosine similarity."""
        return 1.0 - float(self.vectors[i] @ self.vectors[j])

    def parameter_grid(self) -> list[dict]:
        """Only epsilon is tuned."""
        return [{}]

    def _finetune(self, split: Split, texts: list[str]) -> SentenceTransformer:
        """Fine-tune a fresh copy of the base model on the fine-tune part, save it to model_dir and keep its log."""
        model = load_model(self.base_model_dir, self.max_tokens)

        start = time.perf_counter()
        losses = finetune(
            model,
            self.products,
            texts,
            split.finetune,
            seed=self.seed,
            epochs=self.epochs,
            groups_per_batch=self.groups_per_batch,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_fraction=self.warmup_fraction,
            temperature=self.temperature,
        )
        seconds = time.perf_counter() - start

        model.save(str(self.model_dir))
        self.finetune_log = {"epochs": self.epochs, "losses": losses, "seconds": seconds}

        return model

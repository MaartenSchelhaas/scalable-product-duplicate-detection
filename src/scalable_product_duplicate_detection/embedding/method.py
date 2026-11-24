"""
The embedding similarity method: an embedding model, random-hyperplane signatures for
the LSH and the distance
"""

import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from scalable_product_duplicate_detection.bootstrap import Split
from scalable_product_duplicate_detection.data import PreparedProduct, serialize
from scalable_product_duplicate_detection.embedding.finetune import finetune
from scalable_product_duplicate_detection.embedding.hyperplanes import signatures
from scalable_product_duplicate_detection.embedding.model import embed, load_model
from scalable_product_duplicate_detection.similarity_method import SimilarityMethod


class EmbeddingMethod(SimilarityMethod):
    """Embedding model with random-hyperplane LSH and the distance

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
        self.signature_matrix = signatures(self.vectors, self.n_bits, self.seed)

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

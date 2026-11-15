"""
Load an embedding model from a local folder and embed serialized products.
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_model(path: Path, max_tokens: int) -> SentenceTransformer:
    """Load a saved sentence-transformers model that truncates its input after max_tokens tokens.

    Args:
        path (Path): Folder of a saved model
        max_tokens (int): Number of tokens after which input text is truncated

    Returns:
        SentenceTransformer: The model, with the pooling and normalisation stored in its own config
    """
    model = SentenceTransformer(str(path))
    model.max_seq_length = max_tokens

    return model


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Embed texts as unit-length vectors.

    Args:
        model (SentenceTransformer): Model from load_model
        texts (list[str]): Serialized products

    Returns:
        np.ndarray: One row per text (384 columns for bge-small), each of length 1.
    """
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
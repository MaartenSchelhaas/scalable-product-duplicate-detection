"""
File for doing the bootstrapping, for:
bootstrapped sample forming train/test split
and within the train the finetune/validation split for the embedding
"""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class Split:
    train: list[int]
    test: list[int]
    finetune: list[int]
    validation: list[int]


def make_split(model_ids: list[str], finetune_fraction: float, seed: int) -> Split:
    """Draw one bootstrap and split its training part by modelID group.

    Args:
        model_ids (list[str]): modelID of every product, in load_products order
        finetune_fraction (float): Share of the training groups used for fine-tuning; the rest is validation
        seed (int): Seed of this bootstrap

    Returns:
        Split: Indices into the product list for train, test, finetune and validation
    """
    train, test = _draw_bootstrap(len(model_ids), seed)
    finetune, validation = _split_by_group(train, np.array(model_ids), finetune_fraction, seed)

    return Split(
        train=train.tolist(),
        test=test.tolist(),
        finetune=finetune.tolist(),
        validation=validation.tolist(),
    )


def _draw_bootstrap(n_products: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw n_products indices with replacement and deduplicate. The never drawn are test."""
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_products)

    drawn = rng.choice(all_indices, size=n_products, replace=True)
    train = np.unique(drawn)
    test = np.setdiff1d(all_indices, train)

    return train, test


def _split_by_group(train: np.ndarray, model_ids: np.ndarray, finetune_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the training indices into finetune and validation without splitting a modelID group."""
    splitter = GroupShuffleSplit(n_splits=1, train_size=finetune_fraction, random_state=seed)

    # The splitter returns positions in train, not product indices; train[...] turns them into product indices
    finetune_positions, validation_positions = next(splitter.split(train, groups=model_ids[train]))

    return train[finetune_positions], train[validation_positions]
"""
Fine-tune an embedding model on the fine-tune part of one bootstrap with the supervised contrastive (SupCon) loss.
Batches hold whole modelID groups, filled up with same-brand products without a duplicate.
"""

import math
import time
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from transformers import get_linear_schedule_with_warmup

from scalable_product_duplicate_detection.data import PreparedProduct


def finetune(
    model: SentenceTransformer,
    products: list[PreparedProduct],
    texts: list[str],
    finetune_indices: list[int],
    seed: int,
    epochs: int,
    groups_per_batch: int,
    batch_size: int,
    learning_rate: float,
    warmup_fraction: float,
    temperature: float,
) -> list[float]:
    """Train the model in place with SupCon for a fixed number of epochs.

    Args:
        model (SentenceTransformer): Model from embedding.load_model; its weights are updated
        products (list[PreparedProduct]): All products, in load_products order (model_id is the label, brand fills the batches)
        texts (list[str]): Serialized text of every product, same order as products
        finetune_indices (list[int]): Indices of the fine-tune part of this bootstrap
        seed (int): Seed of this bootstrap, for the batches and torch
        epochs (int): Number of epochs; the batches are rebuilt every epoch
        groups_per_batch (int): Number of modelID groups per batch
        batch_size (int): Number of products per batch after filling with products without a duplicate
        learning_rate (float): Peak learning rate of AdamW
        warmup_fraction (float): Share of all steps with linear warmup; linear decay to zero afterwards
        temperature (float): Temperature tau of the SupCon loss

    Returns:
        list[float]: Mean loss per epoch
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # One integer per modelID for the whole run; the loss only compares labels for equality
    _, label_of_product = np.unique([product.model_id for product in products], return_inverse=True)
    groups, singletons = _split_groups_and_singletons(products, finetune_indices)

    # The schedule needs the total number of steps in advance: one step per batch, one batch per groups_per_batch groups
    total_steps = epochs * math.ceil(len(groups) / groups_per_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, round(warmup_fraction * total_steps), total_steps)

    # Training mode switches dropout on
    model.train()

    # Recompute each layer's activations during backward instead of storing all 12 layers:
    # a batch of 48 texts of 512 tokens needs about 2 GB of RAM instead of more than 20 GB, at about 30% more time
    model.gradient_checkpointing_enable()

    epoch_losses = []

    for epoch in range(1, epochs + 1):
        batches = _make_batches(products, groups, singletons, groups_per_batch, batch_size, rng)
        batch_losses = []
        epoch_start = time.perf_counter()

        for number, batch in enumerate(batches, start=1):
            # preprocess returns tensors on the CPU; the model may sit on a GPU
            tokenized = batch_to_device(model.preprocess([texts[index] for index in batch]), model.device)
            embeddings = model(tokenized)["sentence_embedding"]
            labels = torch.tensor(label_of_product[batch], device=model.device)

            loss = _supcon_loss(embeddings, labels, temperature)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            batch_losses.append(loss.item())

            # flush, so a notebook shows each line as soon as the batch is done
            elapsed = time.perf_counter() - epoch_start
            print(f"  epoch {epoch}/{epochs}, batch {number}/{len(batches)}: loss {batch_losses[-1]:.4f}, {elapsed:.0f}s", flush=True)

        epoch_losses.append(sum(batch_losses) / len(batch_losses))
        print(f"epoch {epoch}/{epochs}: mean loss {epoch_losses[-1]:.4f}, {time.perf_counter() - epoch_start:.0f}s", flush=True)

    model.eval()
    return epoch_losses


def _split_groups_and_singletons(products: list[PreparedProduct], indices: list[int]) -> tuple[list[list[int]], list[int]]:
    """Group the indices by modelID: groups have two or more products, singletons have one."""
    members = defaultdict(list)
    for index in indices:
        members[products[index].model_id].append(index)

    groups = [group for group in members.values() if len(group) > 1]
    singletons = [group[0] for group in members.values() if len(group) == 1]

    return groups, singletons


def _make_batches(
    products: list[PreparedProduct],
    groups: list[list[int]],
    singletons: list[int],
    groups_per_batch: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Batches of product indices for one epoch: whole groups sorted by brand, filled with same-brand singletons."""
    ordered_groups = _order_by_brand(products, groups, rng)

    # One shuffled list for all batches of the epoch, so a singleton is used at most once per epoch
    unused = singletons.copy()
    rng.shuffle(unused)

    batches = []
    for start in range(0, len(ordered_groups), groups_per_batch):
        batch_groups = ordered_groups[start : start + groups_per_batch]
        batch = [index for group in batch_groups for index in group]

        fill, unused = _take_fill(products, batch_groups, unused, batch_size - len(batch))
        batches.append(batch + fill)

    rng.shuffle(batches)
    return batches


def _order_by_brand(products: list[PreparedProduct], groups: list[list[int]], rng: np.random.Generator) -> list[list[int]]:
    """Shuffle the groups, then sort them by brand with unknown brand last."""
    shuffled = groups.copy()

    # The shuffle before the stable sort gives the groups of one brand a new order every epoch
    rng.shuffle(shuffled)

    return sorted(shuffled, key=lambda group: (products[group[0]].brand is None, products[group[0]].brand or ""))


def _take_fill(
    products: list[PreparedProduct],
    batch_groups: list[list[int]],
    unused: list[int],
    n_fill: int,
) -> tuple[list[int], list[int]]:
    """Take n_fill singletons from unused, first of the batch's brands, then the next ones; returns (fill, still unused)."""
    brands = {products[group[0]].brand for group in batch_groups} - {None}

    same_brand = [index for index in unused if products[index].brand in brands][:n_fill]
    taken = set(same_brand)
    remaining = [index for index in unused if index not in taken]

    # unused is shuffled, so the front of the list is a random draw
    random_fill = remaining[: n_fill - len(same_brand)]

    return same_brand + random_fill, remaining[len(random_fill) :]


def _supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """SupCon (Khosla et al. 2020) over one batch, averaged over the anchors.

    Row k of embeddings and labels[k] belong to the same product; labels hold the integer modelID per row,
    e.g. ["UN46ES6580", "UN46ES6580", "UN40EH5000"] -> tensor([1, 1, 0]).
    """
    label_list = labels.tolist()
    batch_rows = range(len(label_list))

    # An anchor is a product whose modelID occurs more than once in the batch; the others are only negatives
    anchors = [i for i in batch_rows if label_list.count(label_list[i]) > 1]

    anchor_losses = []
    for i in anchors:
        # A(i): every other product in the batch; P(i): those with the same modelID
        a_of_i = [a for a in batch_rows if a != i]
        p_of_i = [p for p in a_of_i if label_list[p] == label_list[i]]

        scores_i = embeddings @ embeddings[i] / temperature

        log_fractions = scores_i[p_of_i] - torch.logsumexp(scores_i[a_of_i], dim=0)
        anchor_losses.append(-log_fractions.mean())

    # Mean over the anchors instead of Khosla's sum, so the step size does not depend on the number of anchors
    return torch.stack(anchor_losses).mean()
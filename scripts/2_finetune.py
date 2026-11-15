"""
Fine-tune a fresh copy of the base model on the fine-tune part of every bootstrap, then save the model and its training log
"""

import json
import time

from scalable_product_duplicate_detection import config
from scalable_product_duplicate_detection.bootstrap import load_split
from scalable_product_duplicate_detection.data import load_products, prepare, serialize
from scalable_product_duplicate_detection.embedding import load_model
from scalable_product_duplicate_detection.finetune import finetune
from scalable_product_duplicate_detection.seeds import make_seeds

# To time one epoch first: BOOTSTRAPS = [1] and EPOCHS = 1
BOOTSTRAPS = range(1, config.N_BOOTSTRAPS + 1)
EPOCHS = config.FINETUNE_EPOCHS


def main() -> None:
    products = prepare(load_products(config.DATA_PATH), config.BRANDS_PATH)
    texts = [serialize(product) for product in products]

    seeds = make_seeds(config.MASTER_SEED, config.N_BOOTSTRAPS)
    config.FINETUNE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    for number in BOOTSTRAPS:
        split = load_split(config.SPLITS_DIR, number)
        model = load_model(config.BASE_MODEL_DIR, config.MAX_TOKENS)

        start = time.perf_counter()
        losses = finetune(
            model,
            products,
            texts,
            split.finetune,
            seed=seeds[number - 1],
            epochs=EPOCHS,
            groups_per_batch=config.GROUPS_PER_BATCH,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            warmup_fraction=config.WARMUP_FRACTION,
            temperature=config.TEMPERATURE,
        )
        seconds = time.perf_counter() - start

        model.save(str(config.MODELS_DIR / f"bootstrap_{number}"))

        log = {"epochs": EPOCHS, "losses": losses, "seconds": seconds}
        (config.FINETUNE_LOG_DIR / f"bootstrap_{number}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

        print(f"bootstrap {number}: {EPOCHS} epochs in {seconds / 60:.1f} min, final loss {losses[-1]:.4f}")


if __name__ == "__main__":
    main()

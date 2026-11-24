"""
Fine-tune a fresh copy of the base model on the fine-tune part of every bootstrap, then save the model and its training log

for each bootstrap:
    fit the fine-tuned similarity method with retrain on: fine-tune, save to its model folder (overwrites it), embed
    write the training log

Writes the models to config.MODELS_DIR / bootstrap_<number> and the logs to config.FINETUNE_LOG_DIR.
"""

import json

from scalable_product_duplicate_detection import config
from scalable_product_duplicate_detection.bootstrap import load_split
from scalable_product_duplicate_detection.data import load_products, prepare
from scalable_product_duplicate_detection.seeds import make_seeds
from scalable_product_duplicate_detection.similarity_methods import EmbeddingMethod

# To time one epoch first: BOOTSTRAPS = [1] and EPOCHS = 1
BOOTSTRAPS = range(1, config.N_BOOTSTRAPS + 1)
EPOCHS = config.FINETUNE_EPOCHS


def main() -> None:
    products = prepare(load_products(config.DATA_PATH), config.BRANDS_PATH)
    seeds = make_seeds(config.MASTER_SEED, config.N_BOOTSTRAPS)
    config.FINETUNE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    for number in BOOTSTRAPS:
        split = load_split(config.SPLITS_DIR, number)

        similarity_method = EmbeddingMethod(
            products,
            config.EPSILONS,
            base_model_dir=config.BASE_MODEL_DIR,
            model_dir=config.MODELS_DIR / f"bootstrap_{number}",
            fine_tuned=True,
            max_tokens=config.MAX_TOKENS,
            n_bits=config.N_BITS,
            seed=seeds[number - 1],
            epochs=EPOCHS,
            groups_per_batch=config.GROUPS_PER_BATCH,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            warmup_fraction=config.WARMUP_FRACTION,
            temperature=config.TEMPERATURE,
            retrain=True,
        )
        similarity_method.fit(split)

        # With retrain on, fit always fine-tunes and sets the log
        log = similarity_method.finetune_log
        if log is None:
            raise RuntimeError(f"bootstrap {number} was loaded instead of fine-tuned")

        (config.FINETUNE_LOG_DIR / f"bootstrap_{number}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"bootstrap {number}: {EPOCHS} epochs in {log['seconds'] / 60:.1f} min, final loss {log['losses'][-1]:.4f}")


if __name__ == "__main__":
    main()

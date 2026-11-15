"""
Create and save the bootstrap splits used by all later scripts
"""

from scalable_product_duplicate_detection import config
from scalable_product_duplicate_detection.bootstrap import make_split, save_split
from scalable_product_duplicate_detection.data import load_products
from scalable_product_duplicate_detection.seeds import make_seeds


def main() -> None:
    products = load_products(config.DATA_PATH)

    model_ids = []
    for product in products:
        model_ids.append(product.model_id)

    seeds = make_seeds(config.MASTER_SEED, config.N_BOOTSTRAPS)
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for number, seed in enumerate(seeds, start=1):
        split = make_split(model_ids, config.FINETUNE_FRACTION, seed)
        save_split(split, config.SPLITS_DIR, number)

        print(f"bootstrap {number}: train {len(split.train)}, test {len(split.test)}, finetune {len(split.finetune)}, validation {len(split.validation)}")


if __name__ == "__main__":
    main()

"""
Create and save the bootstrap splits used by all later scripts
"""

import json
from dataclasses import asdict

from scalable_product_duplicate_detection.bootstrap import make_split
from scalable_product_duplicate_detection.config import DATA_PATH, FINETUNE_FRACTION, MASTER_SEED, N_BOOTSTRAPS, SPLITS_DIR
from scalable_product_duplicate_detection.data import load_products
from scalable_product_duplicate_detection.seeds import make_seeds


def main() -> None:
    products = load_products(DATA_PATH)

    model_ids = []
    for product in products:
        model_ids.append(product.model_id)

    seeds = make_seeds(MASTER_SEED, N_BOOTSTRAPS)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for number, seed in enumerate(seeds, start=1):
        split = make_split(model_ids, FINETUNE_FRACTION, seed)

        path = SPLITS_DIR / f"bootstrap_{number}.json"
        path.write_text(json.dumps(asdict(split)), encoding="utf-8")

        print(f"bootstrap {number}: train {len(split.train)}, test {len(split.test)}, finetune {len(split.finetune)}, validation {len(split.validation)}")


if __name__ == "__main__":
    main()

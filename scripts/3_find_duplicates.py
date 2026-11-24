"""
Find duplicate TVs in the test part of every bootstrap, with the off-the-shelf and the fine-tuned model

clean all products (once)
for each bootstrap:
    for each run (off the shelf, fine-tuned):
        fit: embed all products, hash them to signatures (once per run)
        for each r of that run:
            tune: LSH on the tuning products, epsilon with the best F1 there
            find_duplicates: LSH on the test products, cluster them with that epsilon
            one row with the LSH and clustering scores of the test part

Writes config.SCORES_PATH with one row per bootstrap, run and r.
"""

import pandas as pd

from scalable_product_duplicate_detection import config
from scalable_product_duplicate_detection.bootstrap import load_split
from scalable_product_duplicate_detection.data import load_products, prepare
from scalable_product_duplicate_detection.embedding.method import EmbeddingMethod
from scalable_product_duplicate_detection.evaluation import cluster_scores, lsh_scores
from scalable_product_duplicate_detection.seeds import make_seeds
from scalable_product_duplicate_detection.similarity_method import SimilarityMethod

BOOTSTRAPS = range(1, config.N_BOOTSTRAPS + 1)

# r = 1 makes every pair a candidate, so the off-the-shelf run is only scored on all comparisons, to show what
# fine-tuning adds; the LSH sweep is for the fine-tuned run
R_VALUES_OF_RUN = {"off_the_shelf": [1], "fine_tuned": config.R_VALUES}

# One of clustering.LINKAGES
LINKAGE = "single"


def main() -> None:
    products = prepare(load_products(config.DATA_PATH), config.BRANDS_PATH)
    seeds = make_seeds(config.MASTER_SEED, config.N_BOOTSTRAPS)
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for number in BOOTSTRAPS:
        split = load_split(config.SPLITS_DIR, number)

        # Same seed, so the same hyperplanes for both runs of a bootstrap
        runs = [
            ("off_the_shelf", config.BASE_MODEL_DIR, False),
            ("fine_tuned", config.MODELS_DIR / f"bootstrap_{number}", True),
        ]

        for run, model_dir, fine_tuned in runs:
            print(f"bootstrap {number} of {config.N_BOOTSTRAPS}, {run}: embedding", flush=True)
            # Saved models are loaded; a missing fine-tuned model is fine-tuned first, with the settings below
            similarity_method = EmbeddingMethod(
                products,
                config.EPSILONS,
                base_model_dir=config.BASE_MODEL_DIR,
                model_dir=model_dir,
                fine_tuned=fine_tuned,
                max_tokens=config.MAX_TOKENS,
                n_bits=config.N_BITS,
                seed=seeds[number - 1],
                epochs=config.FINETUNE_EPOCHS,
                groups_per_batch=config.GROUPS_PER_BATCH,
                batch_size=config.BATCH_SIZE,
                learning_rate=config.LEARNING_RATE,
                warmup_fraction=config.WARMUP_FRACTION,
                temperature=config.TEMPERATURE,
                retrain=False,
            )
            similarity_method.fit(split)

            for rows_per_band in R_VALUES_OF_RUN[run]:
                row = {"bootstrap": number, "run": run}
                row.update(_tune_and_test(similarity_method, split.test, rows_per_band))
                rows.append(row)

                print(f"    r = {rows_per_band:2d}: fraction {row['fraction_of_comparisons']:.4f}, "
                      f"PC {row['pair_completeness']:.3f}, F1 {row['f1']:.3f}", flush=True)

        # Rewritten after every bootstrap, so a crash keeps the bootstraps already done
        pd.DataFrame(rows).to_csv(config.SCORES_PATH, index=False)


def _tune_and_test(similarity_method: SimilarityMethod, test: list[int], rows_per_band: int) -> dict:
    """Tune on the method's tuning part, then the LSH and clustering scores on the test part."""
    similarity_method.tune(rows_per_band, LINKAGE)
    candidates, clusters = similarity_method.find_duplicates(test, rows_per_band, LINKAGE)
    _, epsilon = similarity_method.tuned[(rows_per_band, LINKAGE)]

    lsh = lsh_scores(candidates, similarity_method.products, test)
    final = cluster_scores(clusters, similarity_method.products, test)
    bands = config.N_BITS // rows_per_band

    scores = {
        "r": rows_per_band,
        "b": bands,
        "t": (1 / bands) ** (1 / rows_per_band),
        "fraction_of_comparisons": lsh["fraction_of_comparisons"],
        "pair_quality": lsh["pair_quality"],
        "pair_completeness": lsh["pair_completeness"],
        "f1_star": lsh["f1_star"],
        "epsilon": epsilon,
        "precision": final["precision"],
        "recall": final["recall"],
        "f1": final["f1"],
    }
    return scores


if __name__ == "__main__":
    main()

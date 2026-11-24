"""
Detect duplicate TVs on the test part of every bootstrap, with the off-the-shelf and the fine-tuned model

clean and serialize all products (once)
for each bootstrap:
    for each run (off the shelf, fine-tuned):
        embed all products, hash them to signatures (once per run)
        for each r of that run:
            LSH on the tuning products, epsilon with the best F1 there
            LSH on the test products, cluster them with that epsilon
            one row with the LSH and clustering scores of the test part

Writes config.DETECTION_PATH with one row per bootstrap, run and r.
"""

import numpy as np
import pandas as pd

from scalable_product_duplicate_detection import config
from scalable_product_duplicate_detection.bootstrap import load_split
from scalable_product_duplicate_detection.clustering import best_epsilon, cluster
from scalable_product_duplicate_detection.data import PreparedProduct, load_products, prepare, serialize
from scalable_product_duplicate_detection.embedding import embed, load_model
from scalable_product_duplicate_detection.evaluation import cluster_scores, lsh_scores
from scalable_product_duplicate_detection.lsh import candidate_pairs, signatures
from scalable_product_duplicate_detection.seeds import make_seeds

BOOTSTRAPS = range(1, config.N_BOOTSTRAPS + 1)

# r = 1 makes every pair a candidate, so the off-the-shelf run is only scored on all comparisons, to show what
# fine-tuning adds; the LSH sweep is for the fine-tuned run
R_VALUES_OF_RUN = {"off_the_shelf": [1], "fine_tuned": config.R_VALUES}


def main() -> None:
    products = prepare(load_products(config.DATA_PATH), config.BRANDS_PATH)

    texts = []
    for product in products:
        texts.append(serialize(product))

    seeds = make_seeds(config.MASTER_SEED, config.N_BOOTSTRAPS)
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for number in BOOTSTRAPS:
        split = load_split(config.SPLITS_DIR, number)

        # The fine-tuned model has seen the fine-tune part, so it tunes epsilon on the validation part only
        runs = [
            ("off_the_shelf", config.BASE_MODEL_DIR, split.train),
            ("fine_tuned", config.MODELS_DIR / f"bootstrap_{number}", split.validation),
        ]

        for run, model_dir, tuning in runs:
            print(f"bootstrap {number} of {config.N_BOOTSTRAPS}, {run}: embedding", flush=True)
            vectors = embed(load_model(model_dir, config.MAX_TOKENS), texts)

            # Same hyperplanes for both runs of a bootstrap
            signature_matrix = signatures(vectors, config.N_BITS, seeds[number - 1])

            for rows_per_band in R_VALUES_OF_RUN[run]:
                row = {"bootstrap": number, "run": run}
                row.update(_tune_and_test(vectors, signature_matrix, products, tuning, split.test, rows_per_band))
                rows.append(row)

                print(f"    r = {rows_per_band:2d}: fraction {row['fraction_of_comparisons']:.4f}, "
                      f"PC {row['pair_completeness']:.3f}, F1 {row['f1']:.3f}", flush=True)

        # Rewritten after every bootstrap, so a crash keeps the bootstraps already done
        pd.DataFrame(rows).to_csv(config.DETECTION_PATH, index=False)


def _tune_and_test(
    vectors: np.ndarray,
    signature_matrix: np.ndarray,
    products: list[PreparedProduct],
    tuning: list[int],
    test: list[int],
    rows_per_band: int,
) -> dict:
    """Epsilon with the best F1 on the tuning candidates, then the LSH and clustering scores on the test candidates."""
    tuning_candidates = candidate_pairs(signature_matrix, tuning, rows_per_band)
    clusterings = cluster(vectors, products, tuning, tuning_candidates, config.EPSILONS)
    epsilon = best_epsilon(clusterings, products, tuning)

    test_candidates = candidate_pairs(signature_matrix, test, rows_per_band)
    test_clusters = cluster(vectors, products, test, test_candidates, [epsilon])[epsilon]

    lsh = lsh_scores(test_candidates, products, test)
    final = cluster_scores(test_clusters, products, test)
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

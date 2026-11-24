"""
Show the results of script 3: the mean over the bootstraps per run and r, and the plots of pair quality, pair
completeness, F1* and F1 against the fraction of comparisons, as the assignment asks

Reads config.SCORES_PATH. Writes the table config.MEAN_PER_R_PATH and one figure per x-axis scale (linear as in
the papers, log to spread out the small fractions) to config.FIGURES_DIR.
"""

import matplotlib.pyplot as plt
import pandas as pd

from scalable_product_duplicate_detection import config

TITLE_OF_METRIC = {
    "pair_quality": "Pair quality",
    "pair_completeness": "Pair completeness",
    "f1_star": "F1*",
    "f1": "F1",
}


def main() -> None:
    scores = pd.read_csv(config.SCORES_PATH)

    # The fraction is averaged too, so every r becomes one point of the mean curve
    columns = ["b", "t", "fraction_of_comparisons", "pair_quality", "pair_completeness", "f1_star", "epsilon",
               "precision", "recall", "f1"]
    mean_per_r = scores.groupby(["run", "r"])[columns].mean().reset_index()

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    mean_per_r.to_csv(config.MEAN_PER_R_PATH, index=False)

    fine_tuned = mean_per_r[mean_per_r["run"] == "fine_tuned"]
    off_the_shelf = mean_per_r[mean_per_r["run"] == "off_the_shelf"]

    # r = 1 makes every pair a candidate, so these are the F1 of both runs without LSH
    fine_tuned_all_comparisons = fine_tuned.loc[fine_tuned["r"] == 1, "f1"].item()
    off_the_shelf_all_comparisons = off_the_shelf.loc[off_the_shelf["r"] == 1, "f1"].item()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for scale in ["linear", "log"]:
        figure, axes = plt.subplots(1, 4, figsize=(16, 3.5))

        for axis, (metric, title) in zip(axes, TITLE_OF_METRIC.items()):
            axis.plot(fine_tuned["fraction_of_comparisons"], fine_tuned[metric], marker=".", label="fine-tuned, LSH")
            axis.set_xscale(scale)
            axis.set_xlabel("fraction of comparisons")
            axis.set_title(title)

        f1_axis = axes[-1]
        f1_axis.axhline(fine_tuned_all_comparisons, color="grey", linestyle="--", label="fine-tuned, all comparisons")
        f1_axis.axhline(off_the_shelf_all_comparisons, color="grey", linestyle=":", label="off the shelf, all comparisons")
        f1_axis.legend()

        figure.tight_layout()
        figure.savefig(config.FIGURES_DIR / f"metrics_vs_fraction_{scale}.png", dpi=200)
        plt.close(figure)

    print(f"F1 with all comparisons: off the shelf {off_the_shelf_all_comparisons:.3f}, "
          f"fine-tuned {fine_tuned_all_comparisons:.3f}")
    print(f"table: {config.MEAN_PER_R_PATH}")
    print(f"figures: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()

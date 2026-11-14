"""
Config file for dataclasses
"""

from pathlib import Path

# config.py sits in src/scalable_product_duplicate_detection/, two folders below the repo root
REPO_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_PATH / "data" / "TVs-all-merged.json"
BRANDS_PATH = REPO_PATH / "data" / "brands.csv"
SPLITS_DIR = REPO_PATH / "results" / "splits"

MASTER_SEED = 2026
N_BOOTSTRAPS = 5
FINETUNE_FRACTION = 0.8
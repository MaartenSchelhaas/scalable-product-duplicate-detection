"""
Config file for dataclasses
"""

from pathlib import Path

# config.py sits in src/scalable_product_duplicate_detection/, two folders below the repo root
REPO_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_PATH / "data" / "TVs-all-merged.json"
BRANDS_PATH = REPO_PATH / "data" / "brands.csv"
SPLITS_DIR = REPO_PATH / "results" / "splits"
BASE_MODEL_DIR = REPO_PATH / "models" / "bge-small-en-v1.5"

MASTER_SEED = 2026
N_BOOTSTRAPS = 5
FINETUNE_FRACTION = 0.8

# Hugging Face name, only used by scripts/0_download_model.py; all other code loads BASE_MODEL_DIR
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 512
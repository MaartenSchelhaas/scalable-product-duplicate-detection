"""
Config file for dataclasses
"""

from pathlib import Path

# config.py sits in src/scalable_product_duplicate_detection/, two folders below the repo root
REPO_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_PATH / "data" / "TVs-all-merged.json"
BRANDS_PATH = REPO_PATH / "data" / "brands.csv"
SPLITS_DIR = REPO_PATH / "results" / "splits"
FINETUNE_LOG_DIR = REPO_PATH / "results" / "finetune"
PREDICTIONS_DIR = REPO_PATH / "results" / "predictions"
DETECTION_PATH = PREDICTIONS_DIR / "detection.csv"
TABLES_DIR = REPO_PATH / "results" / "tables"
MEAN_PER_R_PATH = TABLES_DIR / "mean_per_r.csv"
FIGURES_DIR = REPO_PATH / "results" / "figures"
MODELS_DIR = REPO_PATH / "models"
BASE_MODEL_DIR = MODELS_DIR / "bge-small-en-v1.5"

MASTER_SEED = 2026
N_BOOTSTRAPS = 5
FINETUNE_FRACTION = 0.8

# Hugging Face name, only used by scripts/0_download_model.py; all other code loads BASE_MODEL_DIR
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 512

# Fine-tuning
FINETUNE_EPOCHS = 20
GROUPS_PER_BATCH = 10
BATCH_SIZE = 48
LEARNING_RATE = 2e-5
WARMUP_FRACTION = 0.1
TEMPERATURE = 0.07

# LSH: r is the only knob, b = N_BITS // r
N_BITS = 256
R_VALUES = list(range(1, 41))

# Clustering
EPSILONS = [step / 100 for step in range(101)]
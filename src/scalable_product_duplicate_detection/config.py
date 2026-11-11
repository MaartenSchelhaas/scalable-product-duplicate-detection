"""
Config file for dataclasses
"""

from enum import Enum
from pathlib import Path


class SerializationMode(Enum):
    TITLE = "title"
    MODEL_WORDS = "model_words"
    TITLE_KV = "title_and_key_values"


DATA_PATH = Path(r"C:\Users\maart\GitHub\text-mining-and-recommender-systems\scalable-product-duplicate-detection\data\TVs-all-merged.json")
BRANDS_PATH = DATA_PATH.parent / "brands.csv"
SERIALIZATION_MODE = SerializationMode.TITLE_KV

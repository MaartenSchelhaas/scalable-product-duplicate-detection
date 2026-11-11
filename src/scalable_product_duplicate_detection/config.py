"""
Config file for dataclasses
"""

from enum import Enum
from pathlib import Path


class SerializationMode(Enum):
    TITLE = "title"
    MODEL_WORDS = "model_words"
    TITLE_KV = "title_and_key_values"


DATA_PATH = Path("data/TVs-all-merged.json")
SERIALIZATION_MODE = SerializationMode.TITLE_KV

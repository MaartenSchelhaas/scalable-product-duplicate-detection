"""
Config file for dataclasses
"""

from pathlib import Path

DATA_PATH = Path(r"C:\Users\maart\GitHub\text-mining-and-recommender-systems\scalable-product-duplicate-detection\data\TVs-all-merged.json")
BRANDS_PATH = DATA_PATH.parent / "brands.csv"
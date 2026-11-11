"""
File to do the data transformations:
Loading, cleaning, get the brand, and serialize a product to text for embedding
"""

from dataclasses import dataclass
from pathlib import Path

from scalable_product_duplicate_detection.config import SerializationMode


@dataclass(frozen=True)
class Product:
    shop: str
    model_id: str
    title: str
    features: dict[str, str]


@dataclass(frozen=True)
class PreparedProduct:
    shop: str
    model_id: str
    title: str
    features: dict[str, str]
    model_words: list[str]
    brand: str | None


def load_products(path: Path) -> list[Product]: ...


def prepare(products: list[Product]) -> list[PreparedProduct]: ...


def serialize(product: PreparedProduct, mode: SerializationMode) -> str: ...


def _clean_text(text: str) -> str: ...


def _model_words(text: str) -> list[str]: ...


def _build_brand_list(products: list[Product]) -> set[str]: ...


def _extract_brand(product: Product, brands: set[str]) -> str | None: ...
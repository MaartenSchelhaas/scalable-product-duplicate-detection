"""
File to do the data transformations:
Loading, cleaning, get the brand, and serialize a product to text for embedding
"""

import html
import json
import re
import unicodedata
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


def load_products(path: Path) -> list[Product]:
    """Load products information of TVs-all merged

    Args:
        path (Path): Path of the stored TVs-all-merged.json

    Returns:
        list[Product]: List containg all the seperate products within that json
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    products: list[Product] = []

    #Loop over all product ids, and build products for the listing within
    for listings in raw.values():
        for listing in listings:
            products.append(
                Product(
                    shop=listing["shop"],
                    model_id=listing["modelID"],
                    title=listing["title"],
                    features=listing["featuresMap"],
                )
            )
    return products


def prepare(products: list[Product]) -> list[PreparedProduct]: ...


def serialize(product: PreparedProduct, mode: SerializationMode) -> str: ...


def _clean_text(text: str) -> str:
    """Clean one title, attribute key or attribute value."""
    text = _fix_encoding(text)
    text = text.lower()
    text = _remove_shop_names(text)
    text = _normalise_units(text)
    return _tidy(text)


def _fix_encoding(text: str) -> str:
    """Unescape HTML, remove encoding damage and map typographic characters to plain ones."""
    characters_to_remove = ("Â", "\x99")
    characters_to_replace = {"–": "-", "“": '"', "”": '"', "×": "x", "º": "°"}

    text = html.unescape(text)

    # "Â" and "\x99" are encoding damage in the source data ("cd/mÂ²", "Insignia\x99")
    for character in characters_to_remove:
        text = text.replace(character, "")

    # Typographic dashes, quotes, "×" and "º" become the plain "-", '"', "x" and "°"
    for character, replacement in characters_to_replace.items():
        text = text.replace(character, replacement)

    # Look-alike characters become plain ones, e.g. "²" becomes "2"; runs last as it would turn "º" into "o"
    return unicodedata.normalize("NFKC", text)


def _remove_shop_names(text: str) -> str:
    """Remove the shop names that appear in titles and attribute keys."""
    shop_names = ("amazon.com", "bestbuy.com", "best buy", "newegg.com", "thenerds.net")

    for shop_name in shop_names:
        text = text.replace(shop_name, " ")

    return text


def _normalise_units(text: str) -> str:
    """Attach inch and hz to their number as in MSMP+ Table 1: "23 Inch" becomes "23inch"."""
    # A digit, then any spaces or dashes, then an inch spelling that is not part of a longer word
    inch = re.compile(r'(?<=\d)[ -]*(?:inches|inch|")(?![a-z])')

    # The same for the hertz spellings
    hertz = re.compile(r"(?<=\d)[ -]*(?:hertz|hz)(?![a-z])")

    text = inch.sub("inch", text)
    return hertz.sub("hz", text)


def _tidy(text: str) -> str:
    """Collapse repeated whitespace and strip the separators left at both ends."""
    # Removing a shop name leaves a trailing or leading " - " behind
    text = " ".join(text.split())
    return text.strip(" -:,;")


def _model_words(text: str) -> list[str]: ...


def _build_brand_list(products: list[Product]) -> set[str]: ...


def _extract_brand(product: Product, brands: set[str]) -> str | None: ...
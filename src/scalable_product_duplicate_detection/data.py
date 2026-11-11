"""
File to do the data transformations:
Loading, cleaning, get the brand, and serialize a product to text for embedding
"""

import csv
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


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


def prepare(products: list[Product], brands_path: Path) -> list[PreparedProduct]:
    """Clean every product and find its brand.

    Args:
        products (list[Product]): Products as loaded from the data set
        brands_path (Path): Path of the brand list csv

    Returns:
        list[PreparedProduct]: One prepared product per input product
    """
    brands = _build_brand_list(brands_path)
    prepared = []

    for product in products:
        title = _clean_text(product.title)
        features = {_clean_text(key): _clean_text(value) for key, value in product.features.items()}

        prepared.append(
            PreparedProduct(
                shop=product.shop,
                model_id=product.model_id,
                title=title,
                features=features,
                brand=_extract_brand(title, features, brands),
            )
        )

    return prepared


def serialize(product: PreparedProduct) -> str:
    """Turn a prepared product into the text that is embedded.

    Args:
        product (PreparedProduct): A product after cleaning

    Returns:
        str: The title followed by every "key: value" pair, separated by "; "
    """
    parts = [product.title]

    for key, value in product.features.items():
        parts.append(f"{key}: {value}")

    return "; ".join(parts)


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


def _build_brand_list(path: Path) -> set[str]:
    """Read the brand names from the brand list csv."""
    with path.open(encoding="utf-8", newline="") as file:
        return {row["brand"] for row in csv.DictReader(file)}


def _extract_brand(title: str, features: dict[str, str], brands: set[str]) -> str | None:
    """Find the brand in the cleaned title and brand entries; None if the brand is not in the list."""
    # Some products list their brand as a key-value pair, e.g. "brand": "samsung"
    text = " ".join([title, features.get("brand", ""), features.get("brand name", "")])

    found = []
    for brand in brands:
        # Whole words only: "vu" must not match in "laservue", nor "tcl" in the model number "tcl47dt50"
        pattern = rf"(?<![a-z0-9]){re.escape(brand)}(?![a-z0-9])"
        if re.search(pattern, text):
            found.append(brand)

    if len(found) > 1:
        raise ValueError(f"Several brands {found} found in: {text}")

    return found[0] if found else None
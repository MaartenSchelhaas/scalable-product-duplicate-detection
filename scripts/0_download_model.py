"""
Download the off-the-shelf embedding model once and save it in models/, so later scripts load it locally
"""

from sentence_transformers import SentenceTransformer

from scalable_product_duplicate_detection.config import BASE_MODEL_DIR, MODEL_NAME


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    model.save(str(BASE_MODEL_DIR))

    print(f"saved {MODEL_NAME} to {BASE_MODEL_DIR}")


if __name__ == "__main__":
    main()

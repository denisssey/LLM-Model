import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"

# Глобальный кэш, чтобы не перезагружать модель при каждом вызове
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    prefixed = [f"passage: {t}" for t in texts]
    result = get_model().encode(prefixed, normalize_embeddings=True)
    return np.asarray(result)


def embed_query(text: str) -> np.ndarray:
    prefixed = f"query: {text}"
    result = get_model().encode([prefixed], normalize_embeddings=True)
    return np.asarray(result[0])

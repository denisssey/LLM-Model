import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-base"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    prefixed = [f"passage: {t}" for t in texts]
    return get_model().encode(prefixed, normalize_embeddings=True)


def embed_query(text: str) -> np.ndarray:
    prefixed = f"query: {text}"
    result = get_model().encode([prefixed], normalize_embeddings=True)
    return result[0]


if __name__ == "__main__":
    test_texts = [
        "Нужно ли платить за продлёнку в школах?",
        "Как получить путёвку в детский лагерь?",
        "Погода в Москве завтра",
    ]
    vecs = embed_passages(test_texts)
    print("Форма матрицы:", vecs.shape)                # ожидаем (3, 768)
    print("Норма 1-го вектора:", float(np.linalg.norm(vecs[0])))   # ожидаем ~1.0

    q = embed_query("Как оплатить продлёнку?")
    print("Форма запроса:", q.shape)
import json
import logging

import numpy as np

from embeddings import embed_query
from paths import CHUNKS_JSON, EMBEDDINGS_NPY
from schemas import Chunk, RetrievedChunk

logger = logging.getLogger("retrieval")

_embeddings: np.ndarray | None = None
_chunks: list[Chunk] | None = None


def _load_index() -> tuple[np.ndarray, list[Chunk]]:
    global _embeddings, _chunks

    if _embeddings is None:
        _embeddings = np.load(EMBEDDINGS_NPY)
        raw = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
        _chunks = [Chunk(**item) for item in raw]
        logger.info("Загружено чанков: %d", len(_chunks))

    assert _embeddings is not None, "Эмбеддинги не загружены"
    assert _chunks is not None, "Чанки не загружены"
    return _embeddings, _chunks


# Порог подобран путем тестов.
# Релевантные чанки дают 0.82+, чуть понижаем, чтобы не слишком сильно отсекать
def retrieve(
    query: str, top_k: int = 5, minimal_score: float = 0.78
) -> list[RetrievedChunk]:
    embeddings, chunks = _load_index()
    query_vector = embed_query(query)

    # Векторы нормализованы -> скалярное произведение равно косинусной близости
    # Получаем по одному числу на каждый чанк
    scores = embeddings @ query_vector
    top_index = np.argsort(scores)[::-1][:top_k]

    results: list[RetrievedChunk] = []
    for i in top_index:
        score = float(scores[i])
        if score < minimal_score:
            break
        results.append(RetrievedChunk(chunk=chunks[i], score=score))

    if not results:
        logger.warning(
            "Ни один чанк не прошел порог %.2f для запроса: %s", minimal_score, query
        )

    return results

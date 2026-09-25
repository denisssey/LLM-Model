"""Модуль retrieval: поиск релевантных чанков по запросу"""

import json

import numpy as np

from embeddings import embed_query
from paths import CHUNKS_JSON, CHUNK_IDS_JSON, EMBEDDINGS_NPY
from schemas import Chunk, RetrievedChunk


_embeddings: np.ndarray | None = None
_chunks: list[Chunk] | None = None


def _load_index() -> tuple[np.ndarray, list[Chunk]]:
    global _embeddings, _chunks

    if _embeddings is None:
        _embeddings = np.load(EMBEDDINGS_NPY)
        raw = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
        _chunks = [Chunk(**item) for item in raw]
    return _embeddings, _chunks


def retrieve(query: str, top_k: int = 3) -> list[RetrievedChunk]:
    embeddings, chunks = _load_index()
    query_vector = embed_query(query)
    scores = embeddings @ query_vector
    top_index = np.argsort(scores)[::-1][:top_k]

    results: list[RetrievedChunk] = []
    for i in top_index:
        results.append(RetrievedChunk(
            chunk=chunks[i],
            score=float(scores[i]),
        ))
    return results


if __name__ == "__main__":
    test_queries = [
        "Как оплатить продлёнку в школе?",
        "Как получить путёвку в детский лагерь?",
        "Когда подавать заявление в первый класс?",
    ]
    for q in test_queries:
        print(f"\n=== {q} ===")
        for rc in retrieve(q, top_k=3):
            print(f"  [{rc.score:.3f}] {rc.chunk.question}")
import json
import logging

import numpy as np

from embeddings import embed_passages
from logging_config import setup_logging
from paths import CHUNK_IDS_JSON, CHUNKS_JSON, EMBEDDINGS_NPY
from schemas import Chunk

logger = logging.getLogger("build_index")


def build_index() -> None:
    raw = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in raw]
    logger.info("Загружено чанков: %d", len(chunks))

    texts = [f"{c.question} {c.answer}" for c in chunks]

    embeddings = embed_passages(texts)
    logger.info("Эмбеддинги: %s", embeddings.shape)

    # Защита от рассинхрона, если эмбеддингов != чанков,
    # то retrieval будет работать с неправильными парами
    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Количество эмбеддингов: ({embeddings.shape[0]}) "
            f"не совпадает с числом чанков ({len(chunks)})"
        )

    chunk_ids = [c.source_id for c in chunks]

    EMBEDDINGS_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_NPY, embeddings)
    CHUNK_IDS_JSON.write_text(
        json.dumps(chunk_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Сохранил: %s", EMBEDDINGS_NPY)
    logger.info("Сохранил: %s", CHUNK_IDS_JSON)

    loaded_embeddings = np.load(EMBEDDINGS_NPY)
    loaded_ids = json.loads(CHUNK_IDS_JSON.read_text(encoding="utf-8"))

    if loaded_embeddings.shape != embeddings.shape:
        raise ValueError(
            f"Сохраненный файл поврежден, ожидали: {embeddings.shape}, "
            f"получили {loaded_embeddings.shape}"
        )
    if len(loaded_ids) != len(chunks):
        raise ValueError(
            f"chunk_ids.json не совпадает по длине: "
            f"ожидали {len(chunks)}, получили {len(loaded_ids)}"
        )

    logger.info("Проверка: %s | %d id", loaded_embeddings.shape, len(loaded_ids))


if __name__ == "__main__":
    setup_logging()
    build_index()

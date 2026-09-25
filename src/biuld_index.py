import json

import numpy as np

from embeddings import embed_passages
from paths import CHUNKS_JSON, EMBEDDINGS_NPY, CHUNK_IDS_JSON
from schemas import Chunk

def build_index() -> None:
    raw = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in raw]
    print("Загружено чанков:", len(chunks))

    texts = [f"{c.question} {c.answer}" for c in chunks]

    embeddings = embed_passages(texts)
    print("Эмбеддинги:", embeddings.shape)

    chunk_ids = [c.source_id for c in chunks]

    EMBEDDINGS_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_NPY, embeddings)
    CHUNK_IDS_JSON.write_text(
        json.dumps(chunk_ids, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("Сохранил:", EMBEDDINGS_NPY)
    print("Сохранил:", CHUNK_IDS_JSON)

    loaded_embeddings = np.load(EMBEDDINGS_NPY)
    loaded_ids = json.loads(CHUNK_IDS_JSON.read_text(encoding="utf-8"))
    print("Проверка:", loaded_embeddings.shape, "|", len(loaded_ids), "id")

if __name__ == "__main__":
    build_index()
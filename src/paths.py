from pathlib import Path

#paths.py лежит в src/, значит корень проекта на уровня выше (.parent.parent)
BASE_DIRECTORY = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = BASE_DIRECTORY / "data"
RAW_DIRECTORY = DATA_DIRECTORY / "raw"
PROCESSED_DIRECTORY = DATA_DIRECTORY / "processed"

EMBEDDINGS_NPY = BASE_DIRECTORY / "index" / "embeddings.npy"
CHUNK_IDS_JSON = BASE_DIRECTORY / "index" / "chunk_ids.json"

CHUNKS_JSON = PROCESSED_DIRECTORY / "chunks.json"

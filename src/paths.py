from pathlib import Path

# src/paths.py -> parent = src -> parent.parent = корень проекта
BASE_DIRECTORY = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = BASE_DIRECTORY / "data"
RAW_DIRECTORY = DATA_DIRECTORY / "raw"
PROCESSED_DIRECTORY = DATA_DIRECTORY / "processed"

INDEX_DIRECTORY = BASE_DIRECTORY / "index"
EMBEDDINGS_NPY = INDEX_DIRECTORY / "embeddings.npy"
CHUNK_IDS_JSON = INDEX_DIRECTORY / "chunk_ids.json"

CHUNKS_JSON = PROCESSED_DIRECTORY / "chunks.json"
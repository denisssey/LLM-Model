from pathlib import Path

# src/paths.py → parent = src → parent.parent = корень проекта
BASE_DIRECTORY = Path(__file__).resolve().parent.parent

DATA_DIRECTORY = BASE_DIRECTORY / "data"
RAW_DIRECTORY = DATA_DIRECTORY / "raw"
PROCESSED_DIRECTORY = DATA_DIRECTORY / "processed"

CHUNKS_JSON = PROCESSED_DIRECTORY / "chunks.json"
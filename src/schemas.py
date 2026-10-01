from pydantic import BaseModel, Field


class Chunk(BaseModel):
    question: str
    answer: str
    source_id: str
    source_file: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float = Field(..., ge=0.0, le=1.0)
    """Косинусная близость в диапазоне [0, 1]"""


# Конструкция, то что должна ллмка вернуть
class Instruction(BaseModel):
    title: str
    audience: str
    summary: str
    required_documents: list[str]
    steps: list[str]
    deadline: str | None = None
    where_to_apply: str | None = None
    sources: list[str]

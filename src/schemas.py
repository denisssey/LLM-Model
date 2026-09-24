from pydantic import BaseModel, Field

class Chunk(BaseModel):
    question: str = Field(..., description="Текст вопроса")
    """Текст вопроса"""

    answer : str = Field(..., description="Текст ответа")
    """Текст ответа"""

    source_id: str = Field(..., description="ID в исходном html")
    """ID в исходном html"""

    source_file : str = Field(..., description="Имя html файла-источника")
    """Имя html файла-источника"""


class RetrievedChunk(BaseModel):
    chunk: Chunk
    """Найденный чанк"""

    score: float = Field(..., ge=0.0, le=1.0)
    """Косинусная близость в диапазоне [0, 1]"""
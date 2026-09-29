import json

from pydantic import BaseModel, Field

from paths import CHUNKS_JSON


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
    where_to_apply: str
    sources: list[str]


if __name__ == "__main__":
    # Проверка загрузки чанков
    raw = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in raw]
    print("Загружено чанков:", len(chunks))
    print("Первый:", chunks[0].question)
    print()

    # Проверка RetrievedChunk
    rc = RetrievedChunk(chunk=chunks[0], score=0.87)
    print("RetrievedChunk:", rc.chunk.question, "| score =", rc.score)
    print()

    # Проверка, что модель работает
    instr = Instruction(
        title="Получение соцобслуживания для жителя блокадного Ленинграда",
        audience="Жители блокадного Ленинграда",
        summary="Инструкция по оформлению социального обслуживания.",
        required_documents=["Паспорт", "Удостоверение"],
        steps=["Обратиться в МФЦ", "Подать заявление", "Дождаться решения"],
        deadline=None,
        where_to_apply="МФЦ или портал gu.spb.ru",
        sources=[chunks[0].source_id, chunks[1].source_id],
    )
    print("Instruction создана:")
    print("  title:", instr.title)
    print("  steps:", instr.steps)
    print("  sources:", instr.sources)

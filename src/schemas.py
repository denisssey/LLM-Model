import json

from pydantic import BaseModel, Field

from paths import CHUNKS_JSON


class Chunk(BaseModel):
    question: str
    """Текст вопроса"""

    answer: str
    """Текст ответа"""

    source_id: str
    """ID в исходном html"""

    source_file: str
    """Имя html файла-источника"""


class RetrievedChunk(BaseModel):
    chunk: Chunk
    """Найденный чанк"""

    score: float = Field(..., ge=0.0, le=1.0)
    """Косинусная близость в диапазоне [0, 1]"""


class Instruction(BaseModel):
    title: str
    """Заголовок инструкции"""

    audience: str
    """Кому адресована"""

    summary: str
    """Краткое описание услуги"""

    required_documents: list[str]
    """Список документов"""

    steps: list[str]
    """Пошаговый порядок"""

    deadline: str | None = None
    """Сроки (если есть)"""

    where_to_apply: str
    """Куда обращаться"""

    sources: list[str]
    """id чанков источников"""




if __name__ == "__main__":
    # 1. Проверяем загрузку чанков из JSON
    raw = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in raw]
    print("Загружено чанков:", len(chunks))
    print("Первый:", chunks[0].question)
    print()

    # 2. Проверяем RetrievedChunk
    rc = RetrievedChunk(chunk=chunks[0], score=0.87)
    print("RetrievedChunk:", rc.chunk.question, "| score =", rc.score)
    print()

    # 3. Проверяем Instruction — просто создаём экземпляр с фиктивными данными
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


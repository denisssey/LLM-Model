import json

import ollama
from pydantic import ValidationError

from schemas import Instruction, RetrievedChunk


MODEL_NAME = "qwen2.5:3b-instruct"


SYSTEM_PROMPT = """ 
РОЛЬ:
Ты - помощник для составления официальных инструкций 
    для жителей города Санкт-Петербург.

ЗАДАЧА: 
По запросу пользователя составить структурированную инструкцию,
опираясь только на предоставленные фрагменты базы знаний.

ПРАВИЛА: 
1. Используй ТОЛЬКО информацию из предоставленных фрагментов.
2. Отвечай ТОЛЬКО на русском языке.
3. Отвечай ТОЛЬКО валидным JSON без пояснений до и после.
4. Если в предоставленных фрагментах нет информации для ответа, честно напиши в поле summary, что в базе знаний нет подходящей информации.
5. В поле sources перечисли source_id только тех фрагментов, которые ты реально использовал.
6. 6. В поле required_documents включай ТОЛЬКО те документы, которые явно упомянуты в предоставленных фрагментах. Не добавляй по догадке.
7. В поле steps описывай ТОЛЬКО конкретные действия, которые должен сделать пользователь. Не включай в steps справочную информацию о правилах и условиях.

ФОРМАТ ОТВЕТА (JSON):
{
    "title": "заголовок инструкции",
    "audience": "кому адресована (категория граждан)",
    "summary": "краткое описание услуги",
    "required_documents": ["документ 1", "документ 2"],
    "steps": ["шаг 1", "шаг 2"], 
    "deadline": "сроки оказания или null",
    "where_to_apply": "куда обращаться",
    "sources": ["source_id_1", "source_id_2"]
}
ОПИСАНИЕ ПОЛЕЙ:
- title: заголовок инструкции 
- audience: кому адресована (категория граждан)
- summary: краткое описание услуги 
- required_documents: список необходимых документов 
- steps: пошаговый порядок действий
- deadline: сроки оказания или null
- where_to_apply: Куда обращаться 
- sources: список source_id использованных фрагментов
"""

def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, rc in enumerate(chunks, start=1):
        part = (
            f"[Фрагмент {i}] source_id: {rc.chunk.source_id}\n"
            f"Вопрос: {rc.chunk.question}\n"
            f"Ответ: {rc.chunk.answer}"
        )
        parts.append(part)
    return "\n\n".join(parts)


def _build_user_prompt(query: str, context: str) -> str:
    return (
        f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{query}\n\n"
        f"ФРАГМЕНТЫ БАЗЫ ЗНАНИЙ:\n{context}\n\n"
        f"Составь инструкцию в формате JSON согласно системным правилам."
    )


def generate(query: str, chunks: list[RetrievedChunk], max_retries: int = 2) -> Instruction:
    if not chunks:
        raise ValueError(
            f"Не удалось найти информацию в базе данных для вашего запроса."
            "Попробуйте переформулировать."
        )
    context = _build_context(chunks)
    user_prompt = _build_user_prompt(query, context)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        print(f"[generation] Попытка {attempt}/{max_retries}...")

        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            format="json",
            options={"temperature": 0.1},
        )
        content = response["message"]["content"]

        try:
            data = json.loads(content)
            return Instruction(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            print(f"[generation] Невалидный ответ: {e}")
            # Уточняем промпт для повторной попытки
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt
                    + "\n\nВАЖНО: верни ТОЛЬКО валидный JSON. "
                    "Все обязательные поля должны быть заполнены. "
                    "Никаких пояснений до или после JSON.",
                },
            ]

    raise RuntimeError(
        f"Не удалось получить валидный JSON после {max_retries} попыток: {last_error}"
    )


if __name__ == "__main__":
    from retrieval import retrieve

    query = (
        "Какие внеурочные виды деятельности есть для детей?"
    )
    chunks = retrieve(query, top_k=5)

    print(f"\nЗАПРОС: {query}")
    print(f"\nНайдено чанков: {len(chunks)}")
    for rc in chunks:
        print(f"  [{rc.score:.3f}] {rc.chunk.question}")
    print()

    instruction = generate(query, chunks)

    print("=" * 60)
    print("ИНСТРУКЦИЯ")
    print("=" * 60)
    print("Заголовок:       ", instruction.title)
    print("Кому адресована: ", instruction.audience)
    print("Описание:        ", instruction.summary)
    print("Документы:")
    for d in instruction.required_documents:
        print(f"  - {d}")
    print("Шаги:")
    for i, s in enumerate(instruction.steps, start=1):
        print(f"  {i}. {s}")
    print("Сроки:           ", instruction.deadline)
    print("Куда обращаться: ", instruction.where_to_apply)
    print("Источники:       ", instruction.sources)
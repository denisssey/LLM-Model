import logging

import ollama
from pydantic import ValidationError

from schemas import Instruction, RetrievedChunk

logger = logging.getLogger("generation")

MODEL_NAME = "qwen2.5:7b-instruct"


SYSTEM_PROMPT = """РОЛЬ:
Ты - помощник для составления официальных инструкций
    для жителей города Санкт-Петербурга.

ЗАДАЧА:
По запросу пользователя составить структурированную инструкцию,
опираясь только на предоставленные фрагменты базы знаний.

ПРАВИЛА:
1. Используй ТОЛЬКО информацию из предоставленных фрагментов.
2. Отвечай ТОЛЬКО на русском языке.
3. Отвечай ТОЛЬКО валидным JSON без пояснений до и после.
4. Если в предоставленных фрагментах нет информации для ответа, честно напиши в поле summary, что в базе знаний нет подходящей информации.
5. В поле sources перечисли source_id только тех фрагментов, которые ты реально использовал.
6. 6. В поле required_documents включай ТОЛЬКО те документы, которые ДОСЛОВНО упомянуты в предоставленных фрагментах. Если в тексте нет явного упоминания документа — НЕ добавляй его, даже если он кажется логичным. Лучше показать меньше документов, чем выдумать.
7. В поле steps описывай ТОЛЬКО конкретные действия, которые должен сделать пользователь. Не включай в steps справочную информацию о правилах и условиях.
8. Не повторяй одинаковые шаги. Каждый шаг должен быть уникальным.
9. Если для поля deadline или where_to_apply нет информации в фрагментах - укажи null. НЕ выдумывай.
10. Пример JSON показывает только СТРУКТУРУ. Не копируй значения из примера. Заполняй поля реальными данными из фрагментов.
11. Если фрагменты относятся к разным темам, используй ТОЛЬКО те, которые точно соответствуют запросу пользователя. Не смешивай информацию из разных услуг в одном ответе.
12. Если в предоставленных фрагментах нет пошагового процесса — оставь поле steps пустым. НЕ придумывай псевдошаги типа "узнайте" или "понимайте".

ФОРМАТ ОТВЕТА (JSON):
{
    "title": "<заголовок инструкции>",
    "audience": "<кому адресована>",
    "summary": "<краткое описание услуги>",
    "required_documents": ["<документ 1>", "<документ 2>"],
    "steps": ["<шаг 1>", "<шаг 2>"],
    "deadline": "<сроки или null>",
    "where_to_apply": "<куда обращаться или null>",
    "sources": ["<source_id_1>", "<source_id_2>"]
}

ВНИМАНИЕ: поля deadline и where_to_apply должны быть либо строкой, либо null (без кавычек). Не возвращай строку "null" - это ошибка.

ОПИСАНИЕ ПОЛЕЙ:
- title: заголовок инструкции
- audience: кому адресована (категория граждан)
- summary: краткое описание услуги
- required_documents: список необходимых документов
- steps: пошаговый порядок действий
- deadline: сроки оказания или null, если данных нет
- where_to_apply: куда обращаться или null, если данных нет
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


def generate(
    query: str, chunks: list[RetrievedChunk], max_retries: int = 2
) -> Instruction:
    if max_retries <= 0:
        raise ValueError(f"max_retries должен быть > 0, получили {max_retries}")

    if not chunks:
        raise ValueError(
            "Не удалось найти информацию в базе данных для вашего запроса. "
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

        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            format="json",
            options={"temperature": 0.1},
        )
        content = response["message"]["content"]

        try:
            instruction = Instruction.model_validate_json(content)
            logger.info("Валидация прошла успешно")
            return instruction
        except ValidationError as e:
            last_error = e
            logger.warning("Невалидный ответ: %s", e)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt + "\n\nВАЖНО: верни ТОЛЬКО валидный JSON. "
                    "Все обязательные поля должны быть заполнены. "
                    "Никаких пояснений до или после JSON.",
                },
            ]

    raise RuntimeError(
        f"Не удалось получить валидный JSON после {max_retries} попыток: {last_error}"
    )

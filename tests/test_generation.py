import json

import pytest

import generation
from generation import generate
from schemas import Chunk, RetrievedChunk


def make_chunks(n: int = 3) -> list[RetrievedChunk]:
    chunks = []
    for i in range(n):
        chunk = Chunk(
            question=f"Вопрос {i}",
            answer=f"Ответ {i}",
            source_id=f"id-{i}",
            source_file="test.html",
        )
        chunks.append(RetrievedChunk(chunk=chunk, score=0.9))
    return chunks


def make_valid_json(sources: list[str] | None = None) -> str:
    if sources is None:
        sources = ["id-0", "id-1"]
    return json.dumps(
        {
            "title": "Тестовая инструкция",
            "audience": "родители",
            "summary": "краткое описание",
            "required_documents": ["паспорт"],
            "steps": ["шаг 1", "шаг 2"],
            "deadline": "1 апреля",
            "where_to_apply": "МФЦ",
            "sources": sources,
        },
        ensure_ascii=False,
    )


class TestGenerateValidation:
    def test_empty_chunks_raises(self):
        with pytest.raises(ValueError, match="Не удалось найти информацию"):
            generate("запрос", [])

    def test_zero_max_retries_raises(self):
        with pytest.raises(ValueError, match="max_retries должен быть > 0"):
            generate("запрос", make_chunks(), max_retries=0)

    def test_negative_max_retries_raises(self):
        with pytest.raises(ValueError, match="max_retries должен быть > 0"):
            generate("запрос", make_chunks(), max_retries=-1)


class TestGenerateSuccess:
    def test_successful_generation(self, monkeypatch):
        def fake_chat(**kwargs):
            return {"message": {"content": make_valid_json()}}

        monkeypatch.setattr(generation.ollama, "chat", fake_chat)

        instruction = generate("Как подать заявление?", make_chunks())
        assert instruction.title == "Тестовая инструкция"
        assert instruction.sources == ["id-0", "id-1"]

    def test_filters_hallucinated_sources(self, monkeypatch):
        def fake_chat(**kwargs):
            return {
                "message": {
                    "content": make_valid_json(
                        sources=["id-0", "выдуманный-id", "id-1", "ещё-один-фейк"]
                    )
                }
            }

        monkeypatch.setattr(generation.ollama, "chat", fake_chat)

        instruction = generate("запрос", make_chunks())
        assert instruction.sources == ["id-0", "id-1"]

    def test_all_hallucinated_sources(self, monkeypatch):
        def fake_chat(**kwargs):
            return {
                "message": {"content": make_valid_json(sources=["фейк-1", "фейк-2"])}
            }

        monkeypatch.setattr(generation.ollama, "chat", fake_chat)

        instruction = generate("запрос", make_chunks())
        assert instruction.sources == []


class TestGenerateRetry:
    def test_retry_on_invalid_json(self, monkeypatch):
        calls = []

        def fake_chat(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                return {"message": {"content": "это не JSON"}}
            return {"message": {"content": make_valid_json()}}

        monkeypatch.setattr(generation.ollama, "chat", fake_chat)

        instruction = generate("запрос", make_chunks(), max_retries=2)
        assert instruction.title == "Тестовая инструкция"
        assert len(calls) == 2

    def test_retry_exhausted_raises(self, monkeypatch):
        def fake_chat(**kwargs):
            return {"message": {"content": "не JSON"}}

        monkeypatch.setattr(generation.ollama, "chat", fake_chat)

        with pytest.raises(RuntimeError, match="Не удалось получить валидный JSON"):
            generate("запрос", make_chunks(), max_retries=2)

    def test_retry_on_missing_field(self, monkeypatch):
        calls = []

        def fake_chat(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                bad = json.dumps({"audience": "тест"})
                return {"message": {"content": bad}}
            return {"message": {"content": make_valid_json()}}

        monkeypatch.setattr(generation.ollama, "chat", fake_chat)

        instruction = generate("запрос", make_chunks(), max_retries=2)
        assert instruction.title == "Тестовая инструкция"
        assert len(calls) == 2

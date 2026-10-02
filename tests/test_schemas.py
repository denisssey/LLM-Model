import pytest
from pydantic import ValidationError

from schemas import Chunk, Instruction, RetrievedChunk


class TestChunk:
    def test_valid_chunk(self):
        chunk = Chunk(
            question="Как подать заявление?",
            answer="Через портал или МФЦ.",
            source_id="abc-123",
            source_file="test.html",
        )
        assert chunk.question == "Как подать заявление?"
        assert chunk.source_id == "abc-123"

    def test_missing_field_fails(self):
        with pytest.raises(ValidationError):
            Chunk(question="Только вопрос")


class TestRetrievedChunk:
    def test_score_in_range(self):
        chunk = Chunk(question="q", answer="a", source_id="id", source_file="f.html")
        rc = RetrievedChunk(chunk=chunk, score=0.85)
        assert rc.score == 0.85

    def test_score_out_of_range_fails(self):
        chunk = Chunk(question="q", answer="a", source_id="id", source_file="f.html")
        with pytest.raises(ValidationError):
            RetrievedChunk(chunk=chunk, score=1.5)


class TestInstruction:
    def test_valid_instruction(self):
        instr = Instruction(
            title="Тест",
            audience="Тест",
            summary="Тест",
            required_documents=["doc1"],
            steps=["step1"],
            deadline=None,
            where_to_apply=None,
            sources=["id1"],
        )
        assert instr.title == "Тест"
        assert instr.deadline is None

    def test_model_validate_json(self):
        json_str = """
        {
            "title": "Инструкция",
            "audience": "родители",
            "summary": "краткое",
            "required_documents": ["паспорт"],
            "steps": ["шаг 1"],
            "deadline": null,
            "where_to_apply": null,
            "sources": ["abc"]
        }
        """
        instr = Instruction.model_validate_json(json_str)
        assert instr.title == "Инструкция"
        assert instr.required_documents == ["паспорт"]

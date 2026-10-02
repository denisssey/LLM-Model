import pytest

from retrieval import retrieve


class TestRetrievalValidation:
    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="query не может быть пустым"):
            retrieve("")

    def test_whitespace_query_raises(self):
        with pytest.raises(ValueError, match="query не может быть пустым"):
            retrieve("   ")

    def test_zero_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k должен быть > 0"):
            retrieve("Как подать заявление?", top_k=0)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k должен быть > 0"):
            retrieve("Как подать заявление?", top_k=-5)


class TestRetrieveLogic:

    @pytest.fixture
    def mock_index(self, monkeypatch):
        import numpy as np

        import retrieval
        from schemas import Chunk

        chunks = [
            Chunk(
                question=f"Вопрос {i}",
                answer=f"Ответ {i}",
                source_id=f"id-{i}",
                source_file="test.html",
            )
            for i in range(5)
        ]

        embeddings = np.zeros((5, 768), dtype=np.float32)
        for i in range(5):
            embeddings[i, i] = 1.0

        def fake_load_index():
            return embeddings, chunks

        monkeypatch.setattr(retrieval, "_load_index", fake_load_index)
        return embeddings, chunks

    def test_returns_at_most_top_k(self, mock_index, monkeypatch):
        import numpy as np

        import retrieval
        from embeddings import embed_query

        def fake_embed_query(q):
            v = np.zeros(768, dtype=np.float32)
            v[0] = 1.0
            return v

        monkeypatch.setattr(retrieval, "embed_query", fake_embed_query)

        results = retrieval.retrieve("любой запрос", top_k=3, minimal_score=0.0)
        assert len(results) <= 3

    def test_threshold_filters_low_scores(self, mock_index, monkeypatch):
        import numpy as np

        import retrieval

        def fake_embed_query(q):
            v = np.zeros(768, dtype=np.float32)
            v[0] = 1.0
            return v

        monkeypatch.setattr(retrieval, "embed_query", fake_embed_query)

        results = retrieval.retrieve("запрос", top_k=5, minimal_score=0.5)
        assert len(results) == 1
        assert results[0].chunk.source_id == "id-0"

    def test_results_sorted_by_score_desc(self, mock_index, monkeypatch):
        import numpy as np

        import retrieval

        def fake_embed_query(q):
            v = np.zeros(768, dtype=np.float32)
            v[0] = 0.5
            v[1] = 0.866
            return v

        monkeypatch.setattr(retrieval, "embed_query", fake_embed_query)

        results = retrieval.retrieve("запрос", top_k=5, minimal_score=0.0)
        scores = [r.score for r in results]
        assert scores == sorted(
            scores, reverse=True
        ), "Результаты не отсортированы по убыванию"

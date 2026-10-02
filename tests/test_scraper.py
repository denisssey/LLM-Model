from scraper import extract_article_text

VALID_HTML = """
<html>
<body>
<main>
  <article class="accordion">
    <h2 class="visually-hidden" id="q1-title">Как подать заявление?</h2>
    <div class="droppanel__frame">Через портал или МФЦ.</div>
  </article>
  <article class="accordion">
    <h2 class="visually-hidden" id="q2-title">Какие документы нужны?</h2>
    <div class="droppanel__frame">Паспорт и свидетельство о рождении.</div>
  </article>
</main>
</body>
</html>
"""


class TestExtractArticleText:
    def test_extracts_valid_chunks(self):
        chunks = extract_article_text(VALID_HTML, "test.html")
        assert len(chunks) == 2
        assert chunks[0].question == "Как подать заявление?"
        assert chunks[0].answer == "Через портал или МФЦ."
        assert chunks[0].source_id == "q1-title"

    def test_source_file_passed_through(self):
        chunks = extract_article_text(VALID_HTML, "my-file.html")
        assert all(c.source_file == "my-file.html" for c in chunks)

    def test_empty_html_returns_empty(self):
        chunks = extract_article_text("<html></html>", "test.html")
        assert chunks == []

    def test_skips_chunk_without_id(self):
        html = """
        <main>
          <article class="accordion">
            <h2 class="visually-hidden">Без id</h2>
            <div class="droppanel__frame">Ответ.</div>
          </article>
          <article class="accordion">
            <h2 class="visually-hidden" id="ok">С id</h2>
            <div class="droppanel__frame">Ответ.</div>
          </article>
        </main>
        """
        chunks = extract_article_text(html, "test.html")
        assert len(chunks) == 1
        assert chunks[0].source_id == "ok"

    def test_skips_chunk_without_h2(self):
        html = """
        <main>
          <article class="accordion">
            <div class="droppanel__frame">Ответ без вопроса.</div>
          </article>
          <article class="accordion">
            <h2 class="visually-hidden" id="ok">Вопрос</h2>
            <div class="droppanel__frame">Ответ.</div>
          </article>
        </main>
        """
        chunks = extract_article_text(html, "test.html")
        assert len(chunks) == 1
        assert chunks[0].source_id == "ok"

    def test_skips_chunk_without_panel(self):
        html = """
        <main>
          <article class="accordion">
            <h2 class="visually-hidden" id="q1">Вопрос без панели</h2>
          </article>
          <article class="accordion">
            <h2 class="visually-hidden" id="ok">Вопрос</h2>
            <div class="droppanel__frame">Ответ.</div>
          </article>
        </main>
        """
        chunks = extract_article_text(html, "test.html")
        assert len(chunks) == 1
        assert chunks[0].source_id == "ok"

    def test_skips_chunks_outside_main(self):
        html = """
        <body>
          <article class="accordion">
            <h2 class="visually-hidden" id="outside">Вне main</h2>
            <div class="droppanel__frame">Не должен попасть.</div>
          </article>
          <main>
            <article class="accordion">
              <h2 class="visually-hidden" id="inside">Внутри main</h2>
              <div class="droppanel__frame">Ответ.</div>
            </article>
          </main>
        </body>
        """
        chunks = extract_article_text(html, "test.html")
        assert len(chunks) == 1
        assert chunks[0].source_id == "inside"

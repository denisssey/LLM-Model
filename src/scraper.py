import json
from pathlib import Path

from bs4 import BeautifulSoup

from paths import CHUNKS_JSON, RAW_DIRECTORY
from schemas import Chunk


def load_local_html(filepath: Path) -> str:
    with open(filepath, "r", encoding="utf-8-sig") as file:
        return file.read()


def extract_article_text(html: str, source_file: str) -> list[Chunk]:
    soup = BeautifulSoup(html, "lxml")
    chunks: list[Chunk] = []

    for article in soup.select("main article.accordion"):
        h2 = article.select_one("h2.visually-hidden")
        panel = article.select_one(".droppanel__frame")
        if not h2 or not panel:
            continue

        source_id = h2.get("id")
        if not isinstance(source_id, str) or not source_id:
            continue

        chunks.append(
            Chunk(
                question=h2.get_text(strip=True),
                answer=panel.get_text(separator=" ", strip=True),
                source_id=source_id,
                source_file=source_file,
            )
        )

    return chunks


if __name__ == "__main__":
    html_files = sorted(RAW_DIRECTORY.glob("*.html"))
    print("Найдено файлов:", len(html_files))

    all_chunks: list[Chunk] = []
    for filepath in html_files:
        html = load_local_html(filepath)
        all_chunks.extend(extract_article_text(html, filepath.name))

    print("Всего чанков:", len(all_chunks))

    CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_JSON.write_text(
        json.dumps(
            [c.model_dump() for c in all_chunks],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Сохранил:", CHUNKS_JSON)

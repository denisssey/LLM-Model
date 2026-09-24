from bs4 import BeautifulSoup
from pathlib import Path
import json


def load_local_html(filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8-sig") as file:
            return file.read()

def extract_article_text(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    chunks = []

    for article in soup.select("main article.accordion"):
        h2 = article.select_one("h2.visually-hidden")
        panel = article.select_one(".droppanel__frame")

        if not h2 or not panel:
            continue

        chunks.append({
            "question": h2.get_text(strip=True),
            "answer": panel.get_text(separator=" ", strip=True),
            "source_id": h2.get("id"),
        })

    return chunks


if __name__ == "__main__":
    html_files = sorted(Path("data/raw").glob("*.html"))
    print("Найдено файлов:", len(html_files))

    all_chunks = []
    for filepath in html_files:
        html = load_local_html(filepath)
        for chunk in extract_article_text(html):
            chunk["source_file"] = filepath.name
            all_chunks.append(chunk)

    print("Всего чанков", len(all_chunks))

    out_direcotry = Path("data/processed")

    out_path = out_direcotry / "chunks.json"
    out_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Сохранил:", out_path)


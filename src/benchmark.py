"""Сравнение скорости моделей qwen2.5:3b и qwen2.5:7b."""

import time

import ollama


PROMPT = "Составь короткую инструкцию из 3 шагов по подаче заявления в школу. Ответь JSON."
MODELS = ["qwen2.5:3b-instruct", "qwen2.5:7b-instruct"]
RUNS = 3    # сколько раз замерять каждую модель


def measure(model: str) -> None:
    """Прогнать модель N раз и вывести метрики."""
    print(f"\n=== {model} ===")

    # Прогрев — первый запрос грузит модель в память, не считаем его
    print("Прогрев...")
    ollama.chat(model=model, messages=[{"role": "user", "content": "тест"}])

    times = []
    tok_per_sec = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        r = ollama.chat(model=model, messages=[{"role": "user", "content": PROMPT}])
        elapsed = time.perf_counter() - t0

        tok_s = r["eval_count"] / (r["eval_duration"] / 1e9)
        times.append(elapsed)
        tok_per_sec.append(tok_s)

        print(f"  Прогон {i + 1}: {elapsed:.2f} сек | {r['eval_count']} токенов | {tok_s:.1f} tok/s")

    print(f"  Среднее: {sum(times) / len(times):.2f} сек | {sum(tok_per_sec) / len(tok_per_sec):.1f} tok/s")


if __name__ == "__main__":
    for model in MODELS:
        measure(model)
# RAG-сервис генерации инструкций.
Прототип сервиса, с использованием RAG-архитектуры, которая опирается на базу знаний портала "Госуслуги СПБ", для составления пошаговых инструкций руководств по запросу пользователя.

## Стек проекта
* Python 3.12
* BeautifulSoup
* lxml
* numpy
* pydantic
* Ollama
* multilingual-e5-base
* qwen2.5:7b-instruct

## Установка
1. Клонирование репозитория 
```bash
git clone https://github.com/denisssey/LLM-Model 
cd "ТЗ. ИАЦ СПБ" 
```
2. Установка зависимостей
```bash
pip install -r requirements.txt
```
3. Установка Ollama и модели \
Скачать Ollama с ollama.com/download и установить.

    Скачать модель LLM:
```bash 
ollama pull qwen2.5:7b-instruct
```
## Запуск
1. Сборка чанков
```bash
python src/scraper.py
```
2. Построение индекса
```bash
python src/build_index.py
```

3. Запуск **demo**
```bash
notebook/demo.ipynb
```
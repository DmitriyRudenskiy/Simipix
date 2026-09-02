# Локальные веса модели

Здесь лежат веса **DINOv2-giant** (`facebook/dinov2-giant`) для работы **без
интернета** (offline-режим). Папка пустая — её нужно один раз заполнить.

## Загрузить модель

```bash
# в корне проекта:
python download_model.py --out ./models
```

Скачает ~2.5 ГБ и положит в `./models` три файла:

- `config.json` — конфиг модели,
- `preprocessor_config.json` — конфиг предобработки,
- `model.safetensors` — веса (1536-мерные эмбеддинги).

Первый раз нужна сеть (загрузка с HuggingFace Hub). Можно указать свою
диркторию или репозиторий:

```bash
python download_model.py --out ./models --model facebook/dinov2-large
```

## Использовать локальные веса

Передайте `--model-dir ./models` в командах:

```bash
imgsim index ./photos --db ./image_db --model-dir ./models
imgsim search ./sample.jpg --db ./image_db --model-dir ./models --top 50
imgsim browse --db ./image_db --model-dir ./models
```

Если `--model-dir` не передан, веса грузятся с HuggingFace Hub по
`facebook/dinov2-giant`.

> Папка в репозитории пустая — веса большие и личные, они не хранятся в git
> (см. `.gitignore`). Здесь только `.gitkeep` и эта инструкция.

#!/usr/bin/env python3
"""Загрузить веса DINOv2-giant локально в models/ для офлайн-использования.

После этого imgsim может грузить модель с диска:
    imgsim index ./photos --db ./image_db --model-dir ./models

Или укажите произвольную директорию/репозиторий:
    python download_model.py --out ./models
    python download_model.py --model facebook/dinov2-large --out ./models
"""

import argparse
from pathlib import Path

from imgsim import config


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Загрузить веса DINOv2-giant локально в models/.")
    ap.add_argument("--model", default=config.MODEL_ID,
                    help=f"репозиторий DINOv2 (по умолчанию {config.MODEL_ID})")
    ap.add_argument("--out", type=Path, default=config.MODELS_DIR,
                    help=f"локальная директория (по умолчанию {config.MODELS_DIR})")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Модель: {args.model}")
    print(f"Папка:  {args.out.resolve()}")

    from transformers import AutoImageProcessor, AutoModel

    print("Загружаю процессор...")
    AutoImageProcessor.from_pretrained(args.model).save_pretrained(args.out)
    print("Загружаю веса модели...")
    AutoModel.from_pretrained(args.model).save_pretrained(args.out)

    files = sorted(p.name for p in args.out.iterdir())
    print(f"Готово. В {args.out}:\n  " + "\n  ".join(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

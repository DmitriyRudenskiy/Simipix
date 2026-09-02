"""Дополнительный анализ изображений: цветовая палитра и тип контента.

- ``palette`` — доминантные цвета (чистый PIL, без тяжёлых зависимостей).
- ``content`` — классификация safe/nsfw моделью ``Falconsai/nsfw_image_detection``.

Ленивый загрузчик модели потокобезопасен (модель грузится один раз, как в
``pose.py``). Ошибки анализа не ломают индексацию — возвращается пустой/нейтральный
результат.
"""
from __future__ import annotations

import threading

import numpy as np
from PIL import Image

# ponytail: цвет в RGB не перцептуально равномерен; для более точного
# «похожести палитр» нужен Lab/OKLab — но это дороже и пока не востребовано.
PALETTE_COLOR_COUNT = 20  # доманантных цветов в палитре и в векторе для поиска


def palette(img: Image.Image, color_count: int = PALETTE_COLOR_COUNT) -> list[dict]:
    """Домантные цвета изображения: [{hex, rgb:[r,g,b], percent}]."""
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    q = img.quantize(colors=color_count, method=2, dither=0)
    pal = q.getpalette()
    total = img.size[0] * img.size[1]
    out: list[dict] = []
    for count, idx in q.getcolors() or []:
        r, g, b = pal[idx * 3], pal[idx * 3 + 1], pal[idx * 3 + 2]
        out.append({
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "rgb": [r, g, b],
            "percent": round(count / total * 100, 2),
        })
    out.sort(key=lambda x: x["percent"], reverse=True)
    return out


def palette_vector(pal: list[dict], color_count: int = PALETTE_COLOR_COUNT) -> list[float]:
    """Вектор палитры для cosine-поиска: до N цветов, rgb×255 взвешены по доле.

    Длина = color_count*3. Вектор нормируется в ``flat_cosine``/поиске.
    """
    vec: list[float] = []
    for item in pal[:color_count]:
        w = item["percent"] / 100.0
        r, g, b = item["rgb"]
        vec += [r / 255.0 * w, g / 255.0 * w, b / 255.0 * w]
    vec += [0.0] * (color_count * 3 - len(vec))
    return vec


class ContentDetector:
    """Ленивый загрузчик NSFW-классификатора (CPU-safe, потокобезопасный)."""

    def __init__(self) -> None:
        self._pipe = None
        self._lock = threading.Lock()

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            with self._lock:
                if self._pipe is None:
                    self._pipe = pipeline(
                        "image-classification",
                        model="Falconsai/nsfw_image_detection",
                        device=-1,  # CPU: на этой машине нет CUDA
                    )
        return self._pipe

    def detect(self, img: Image.Image) -> dict:
        """{'label':'safe'|'nsfw'|'unknown','safe':..,'nsfw':..}."""
        try:
            out = self._load()(img.convert("RGB"))
        except Exception:
            return {"label": "unknown", "safe": 0.0, "nsfw": 0.0}
        safe = nsfw = 0.0
        for o in out:
            label = o["label"].lower()
            score = float(o["score"])
            if any(x in label for x in ("nsfw", "porn", "sexy", "hentai")):
                nsfw = max(nsfw, score)
            else:
                safe = max(safe, score)
        label = "nsfw" if nsfw > safe else "safe"
        return {"label": label, "safe": round(safe, 4), "nsfw": round(nsfw, 4)}


# Единый на модуль детектор: модель грузится один раз и кэшируется (лениво,
# потокобезопасно). detect_content вызывается в индексации по одному на картинку,
# поэтому новый ContentDetector на каждый вызов заново грузил бы модель с диска
# — это была бы серьёзная потеря скорости, поэтому детектор общий.
_detector = ContentDetector()


def detect_content(img: Image.Image) -> dict:
    return _detector.detect(img)

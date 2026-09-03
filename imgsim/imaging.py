"""Работа с изображениями: поиск файлов, быстрое сжатие, миниатюры, хэши.

Ключевые решения для БОЛЬШИх изображений:
  1. draft-декод JPEG — файл декодируется сразу в уменьшенном разрешении
     (быстро и экономно по памяти), полноразмерное декодирование не выполняется.
  2. Перед векторизацией картинка приводится к max_side (по умолчанию 1024 px) —
     процессор DINOv2 всё равно режет её до 224x224.
  3. Миниатюра для галереи — минимальная WebP (320 px, q78), один файл на
     уникальный контент, имя = sha1 контента (естественная дедупликация).
"""

import hashlib
import io
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps

from . import config

# Разрешаем очень большие изображения; размер файла ограничиваем сами
# (config.MAX_FILE_BYTES), чтобы не падать на аномалиях.
Image.MAX_IMAGE_PIXELS = 200_000_000


def _resize_to_max(img: Image.Image, max_side: int) -> Image.Image:
    """Ресайз по большей стороне до max_side (BILINEAR), если больше."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                      Image.BILINEAR)


def iter_image_files(root: Path, recursive: bool = True,
                     exclude_dirs: tuple = ()) -> Iterator[Path]:
    """Итератор по файлам-изображениям внутри root.

    Пропускает скрытые каталоги (начинающиеся с '.') и папки из exclude_dirs.
    """
    it = root.rglob("*") if recursive else root.glob("*")
    excl = {Path(e).resolve() for e in exclude_dirs}
    for p in sorted(it):
        try:
            if not p.is_file() or p.suffix.lower() not in config.IMAGE_EXTS:
                continue
            rel = p.relative_to(root)
            if any(part.startswith(".") for part in rel.parts[:-1]):
                continue  # скрытые каталоги (.git, кэши и т.п.)
            resolved = p.resolve()
            if any(resolved == e or e in resolved.parents for e in excl):
                continue
            yield p
        except OSError:
            continue


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def prepare_image(data: bytes, max_side: int = config.PREPROCESS_MAX_SIDE) -> Image.Image:
    """bytes -> RGB PIL, уменьшенная до max_side по большей стороне.

    Используется перед векторизацией: экономит память и время на больших фото.
    """
    img = Image.open(io.BytesIO(data))
    if img.format == "JPEG":
        # Быстрый декод сразу в уменьшенном размере
        img.draft("RGB", (max_side, max_side))
    img = ImageOps.exif_transpose(img)  # учитываем ориентацию из EXIF
    img = img.convert("RGB")
    return _resize_to_max(img, max_side)


def make_thumbnail(data: bytes, thumbs_dir: Path, sha: str,
                   size: int = config.THUMB_MAX_SIDE,
                   quality: int = config.THUMB_QUALITY) -> Path:
    """Создаёт (или переиспользует) минимальную миниатюру WebP.

    Возвращает абсолютный путь к файлу миниатюры. Имя = sha1 контента,
    поэтому одинаковые картинки не дублируются на диске.
    """
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    out = thumbs_dir / f"{sha[:20]}.webp"
    if out.exists() and out.stat().st_size > 0:
        return out
    img = Image.open(io.BytesIO(data))
    if img.format == "JPEG":
        img.draft("RGB", (size, size))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = _resize_to_max(img, size)
    tmp = out.with_suffix(".tmp")
    img.save(tmp, "WEBP", quality=quality, method=5)
    tmp.replace(out)
    return out

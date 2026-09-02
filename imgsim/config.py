"""Общие константы и настройки imgsim."""

import os
from pathlib import Path

# Корень проекта (папка выше imgsim/). Отсюда берём локальные модели и ./models.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_NAME = "imgsim"
VERSION = "1.0.0"

# --- Модель -----------------------------------------------------------------
# Всегда используется самая большая модель — максимальное качество поиска.
MODEL = "giant"
MODEL_ID = "facebook/dinov2-giant"
MODEL_DIM = 1536

# Локальная директория с весами модели. Пусто → грузим с HuggingFace Hub.
# Заполнить: python download_model.py
MODELS_DIR = PROJECT_ROOT / "models"

# Весы детектора лиц (yolov8n-face). Точку нахождения задаём здесь, а не в
# pose.py: переопределить через env IMGSIM_FACE_MODEL, проект станет переносимым.
# Дефолт — models/ (туда же кладутся и веса модели); ради обратной
# совместимости с прежней раскладкой падает на старый путь, если модели в
# models/ пока нет.
_DEFAULT_FACE_MODEL = MODELS_DIR / "yolov8n-face.pt"
_LEGACY_FACE_MODEL = Path(
    "/Users/user/PycharmProjects/FaceTools/models/yolov8n-face.pt")
_face_env = os.environ.get("IMGSIM_FACE_MODEL")
FACE_MODEL = (Path(_face_env) if _face_env
              else (_DEFAULT_FACE_MODEL
                    if _DEFAULT_FACE_MODEL.exists()
                    else _LEGACY_FACE_MODEL))

# --- Изображения --------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".gif",
              ".tif", ".tiff"}

# Сжатие перед векторизацией: JPEG декодируется сразу в уменьшенном размере
# (draft-режим), затем ресайз до этой стороны. Процессор DINOv2 всё равно
# приводит картинку к 224x224, поэтому большего не требуется.
PREPROCESS_MAX_SIDE = 1024

# Минимальная миниатюра для галереи: хранится в <db>/thumbs один раз на
# уникальный контент (имя = sha1 контента), формат WebP.
THUMB_MAX_SIDE = 320
THUMB_QUALITY = 78

# Защита от аномально огромных файлов (байты)
MAX_FILE_BYTES = 512 * 1024 * 1024  # 512 МБ

# --- LanceDB ------------------------------------------------------------------
TABLE_PREFIX = "images_"          # images_giant
# IVF-PQ индекс строим, начиная с такого числа строк; до этого — точный
# плоский поиск (он быстрее на малых объёмах).
INDEX_MIN_ROWS = 100_000
# Сбрасывать буфер накопленных строк в базу каждые N строк (меньше сегментов)
STORE_FLUSH_ROWS = 64  # ponytail: пачка в RAM перед эмбеддом; ~640 МБ PIL + пачка эмбедда. 256 проседал в swap на загруженной машине — замедлял индексацию; 64 устойчивее

DEFAULT_DB_DIR = "./image_db"
DEFAULT_RESULTS_DIR = "./results"

# Страница каталога (`browse`) — сколько записей встраивать в HTML с векторами
# для клиентского cosine-поиска. До этого порога — текстовый фильтр + похожие.
# ponytail: вектора в HTML = ~dim*8 байт/запись; на больших каталогах HTML
# раздувается — тогда нужен серверный поиск, а не самодосточный файл.
BROWSE_MAX_EMBED = 300  # ponytail: >300 записей — только текстовый поиск (встраивание 1536-мерных вектора даёт HTML по 80+ МБ, тяжело для браузера); визуальный поиск — в search/find_duplicates


def default_batch_size(device: str) -> int:
    return 16 if device == "cuda" else 4


def results_dir() -> Path:
    return Path(DEFAULT_RESULTS_DIR)

"""Файловая блокировка во время индексации.

LanceDB не любит, когда параллельно пишут/читают ту же таблицу, поэтому
индексация ставит замок на каталог базы, а index/search/browse/delete его
проверяют и отказывают, если инукс занят.
"""

import os
import time
from pathlib import Path

LOCK_NAME = ".indexing"
# если лок-файлу больше этого — процесс уронился и не почистил, считаем мёртвым
STALE_SECONDS = 3600


def _path(db_dir) -> Path:
    return Path(db_dir) / LOCK_NAME


def is_locking(db_dir) -> bool:
    p = _path(db_dir)
    if not p.exists():
        return False
    try:
        age = time.time() - p.stat().st_mtime
    except OSError:
        return False
    return age <= STALE_SECONDS


def acquire(db_dir) -> None:
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    _path(db_dir).write_text(f"pid={os.getpid()}\nstart={time.time()}\n")


def release(db_dir) -> None:
    try:
        _path(db_dir).unlink()
    except OSError:
        pass

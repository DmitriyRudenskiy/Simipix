"""Проверка дедупликации по хешу и защиты от перемещения.

lancedb/torch могут отсутствовать, поэтому подменяем embedder и store на
фейки, но гоняем РЕАЛЬНЫЙ код indexer.run_index.

Сценарии:
  1. N уникальных -> добавлено N, дублей 0.
  2. Дубль контента в папке -> пропущен, дубля в базе нет.
  3. Переименование/перемещение -> запись обновлена, дубля нет.
  4. Удаление файла + --prune -> запись из базы удалена.
"""

import io
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import imgsim.indexer as indexer
from imgsim import config


class FakeEmbedder:
    def __init__(self, model_dir=None, log=print):
        self.device = "cpu"

    def embed_with_retry(self, images, log=print):
        return np.random.default_rng(0).random((len(images), config.MODEL_DIM),
                                               dtype=np.float32)


class FakeStore:
    """Контракт ImageStore, нужный indexer'у. Состояние на диске (pkl) по
    ключу db_dir — как у реальной базы, чтобы «из сценов» видели данные
    предыдущих индексаций."""

    _cache = {}

    def __init__(self, db_dir, model):
        self.db_dir = Path(db_dir)
        self.thumbs_dir = self.db_dir / "thumbs"
        self.model = model
        self.table_name = "images_" + model
        key = str(self.db_dir)
        if key in FakeStore._cache:
            s = FakeStore._cache[key]
            self.rows, self.db_hashes = s.rows, s.db_hashes
        else:
            self.rows, self.db_hashes = [], set()

    def _save(self):
        FakeStore._cache[str(self.db_dir)] = self

    def existing_hashes(self):
        return set(self.db_hashes)

    def get_meta(self, h):
        for r in self.rows:
            if r["hash"] == h:
                return dict(r)
        return None

    def add_rows(self, rows):
        for r in rows:
            self.rows.append(dict(r))
            self.db_hashes.add(r["hash"])
        self._save()

    def upsert_rows(self, rows):
        n = 0
        for r in rows:
            if r["hash"] in self.db_hashes:
                row = next(x for x in self.rows if x["hash"] == r["hash"])
                row.update(path=r["path"], mtime=r["mtime"],
                           size=r["size"], thumb=r["thumb"])
                n += 1
        self._save()
        return n

    def prune_missing_hashes(self, keep):
        before = len(self.rows)
        self.rows = [r for r in self.rows if r["hash"] in keep]
        self.db_hashes = {r["hash"] for r in self.rows}
        self._save()
        return before - len(self.rows)

    def maybe_create_ann_index(self, log=print):
        return False

    def has_vector_index(self):
        return False

    def optimize(self, log=print):
        pass

    def rows_count(self):
        return len(self.rows)

    def drop(self):
        self.rows.clear()
        self.db_hashes.clear()
        FakeStore._cache.pop(str(self.db_dir), None)


def _write(dirpath, name, content):
    p = dirpath / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _img_bytes(color, size=(12, 12)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def run():
    tmp = Path(tempfile.mkdtemp(prefix="imgsim_test_"))
    try:
        db = tmp / "db"
        src = tmp / "photos"
        a, b, c = (_img_bytes((200, 30, 30)),
                   _img_bytes((30, 200, 30)),
                   _img_bytes((30, 30, 200)))
        _write(src, "a.png", a)
        _write(src, "b.png", b)
        _write(src, "c.png", c)

        orig_e, orig_s = indexer.DINOv2Embedder, indexer.ImageStore
        indexer.DINOv2Embedder, indexer.ImageStore = FakeEmbedder, FakeStore
        try:
            s1 = indexer.run_index(str(src), str(db), log=lambda *a: None)
            assert s1["added"] == 3 and s1["duplicates"] == 0 \
                and s1["total_rows"] == 3, s1
            print("1 уникальные: OK", s1["added"], "записей")

            _write(src, "a_dup.png", a)
            s2 = indexer.run_index(str(src), str(db), log=lambda *a: None)
            assert s2["duplicates"] >= 1 and s2["added"] == 0 \
                and s2["total_rows"] == 3, s2
            print("2 дубль контента: OK", "dup=", s2["duplicates"],
                  "total=", s2["total_rows"])

            src.joinpath("a.png").unlink()
            _write(src, "renamed.png", a)
            s3 = indexer.run_index(str(src), str(db), log=lambda *a: None)
            assert s3["updated"] == 1 and s3["added"] == 0 \
                and s3["total_rows"] == 3, s3
            print("3 перемещение: OK", "updated=", s3["updated"],
                  "total=", s3["total_rows"])

            src.joinpath("b.png").unlink()
            s4 = indexer.run_index(str(src), str(db), prune=True,
                                   log=lambda *a: None)
            assert s4["total_rows"] == 2, s4
            print("4 prune: OK", "total=", s4["total_rows"],
                  "removed=", s4.get("removed"))
        finally:
            indexer.DINOv2Embedder, indexer.ImageStore = orig_e, orig_s
        print("\nВсе проверки пройдены.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(run())

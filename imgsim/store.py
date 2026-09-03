"""Слой хранения: LanceDB (векторная таблица) + каталог миниатюр.

Записи хранится по ХЕШУ содержимого файла, а не по пути. Поэтому переименование
и перемещение изображения обновляют существующую запись (путь, mtime, миниатюру),
не создавая дубля — защита от дублирования по перемещению.

- Одна таблица на giant: images_giant.
- До INDEX_MIN_ROWS строк — точный плоский поиск; после — автоматически
  создаётся IVF-PQ индекс (cosine).
"""

import math
import re
from pathlib import Path

from . import config

# sha1 hex — единственный легальный формат хеша-первичного ключа.
_HASH_RE = re.compile(r"^[0-9a-f]{40}$")


def _check_hash(h: str) -> str:
    """sha1-хеш (40 hex): проверяем один раз, а не в каждом escape-блоке.

    Защишает where/delete от инъекций — на практике хеши всегда легальны, но
    держать проверку рядом с формированием запроса дешевле, чем полагаться на
    «везде только hex». Хеш из файла/базы не может несено вбросить quote."""
    if not isinstance(h, str) or not _HASH_RE.match(h):
        raise ValueError(f"некорректный хеш: {h!r}")
    return h


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class ImageStore:
    def __init__(self, db_dir, model: str = config.MODEL):
        assert model == config.MODEL, f"imgsim работает только с {config.MODEL}"
        self.model = model
        self.db_dir = Path(db_dir)
        self.dim = config.MODEL_DIM
        self.table_name = config.TABLE_PREFIX + model
        self.thumbs_dir = self.db_dir / "thumbs"

        import lancedb
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_dir))
        self._table = None

    # ------------------------------------------------------------------ schema
    def schema(self):
        import pyarrow as pa
        return pa.schema([
            ("hash", pa.string()),        # PRIMARY KEY: sha1 содержимого
            ("path", pa.string()),        # последний известный путь
            ("mtime", pa.float64()),      # время модификации
            ("size", pa.int64()),         # размер файла, байт
            ("thumb", pa.string()),       # путь к миниатюре WebP
            ("vector", pa.list_(pa.float32(), self.dim)),
            ("pose", pa.string()),        # JSON нормализованной позы (DW-Pose)
            ("face_vector", pa.list_(pa.float32(), self.dim)),  # эмбеддинг лица (похожие люди)
            ("palette", pa.string()),     # JSON домантных цветов (hex/rgb/percent)
            ("content", pa.string()),     # JSON safe/nsfw классификации контента
        ])

    # ------------------------------------------------------------------- table
    def table(self, create: bool = True):
        if self._table is None:
            if self.table_name in self.db.table_names():
                self._table = self.db.open_table(self.table_name)
            elif create:
                self._table = self.db.create_table(
                    self.table_name, schema=self.schema())
            else:
                return None
        return self._table

    def exists(self) -> bool:
        return self.table_name in self.db.table_names()

    def rows_count(self) -> int:
        t = self.table(create=False)
        return t.count_rows() if t is not None else 0

    def drop(self) -> None:
        if self.exists():
            self.db.drop_table(self.table_name)
        self._table = None

    # ------------------------------------------------------------- by hash -----
    def find_hashes(self, hashes: list) -> set:
        """Возвращает подсет хешей, которые уже есть в таблице."""
        t = self.table(create=False)
        if t is None or not hashes:
            return set()
        # Хеш — это sha1 (40 символов): читаем только столбец hash и
        # фильтруем в Python (в этой версии lancedb/pyarrow .where со строкой
        # не поддерживается ни на таблице, ни на arrow).
        arrow = t.to_arrow().select(["hash"])
        existing = set(arrow.column("hash").to_pylist())
        return set(hashes) & existing

    def get_meta(self, hash: str, with_vector: bool = False) -> dict | None:
        """Путь/mtime/size/миниатюра/поза/face_vector по хешу (None, если хеша нет).

        with_vector=True — добавляет столбец vector (нужен для image-поиска по
        сохранённому вектору запроса, без пере-эмбеддинга)."""
        t = self.table(create=False)
        if t is None:
            return None
        cols = ["hash", "path", "mtime", "size", "thumb", "pose", "face_vector",
                "palette", "content"]
        if with_vector:
            cols = ["hash", "path", "mtime", "size", "thumb", "vector",
                    "face_vector", "palette", "content"]
        arrow = t.to_arrow().select(cols)
        for r in arrow.to_pylist():
            if r.get("hash") == hash:
                return r
        return None

    def existing_hashes(self) -> set:
        """Все хеши в таблице (для дедупликации при индексации)."""
        t = self.table(create=False)
        if t is None or t.count_rows() == 0:
            return set()
        try:
            arrow = t.to_lance().to_table(columns=["hash"])
        except Exception:
            arrow = t.to_arrow().select(["hash"])
        return set(arrow.column("hash").to_pylist())

    # --------------------------------------------------------------------- add
    def add_rows(self, rows: list) -> None:
        if not rows:
            return
        import pyarrow as pa
        t = self.table()
        t.add(pa.Table.from_pylist(rows, schema=self.schema()))

    def upsert_rows(self, rows: list) -> int:
        """Добавить новые записи и обновить (path/mtime/size/thumb) у существующих.

        Возвращает число обновлённых записей (перемещённых/переимёванных).
        """
        if not rows:
            return 0
        existing = self.find_hashes([r["hash"] for r in rows])
        n_upd = sum(1 for r in rows if r["hash"] in existing)
        import pyarrow as pa
        t = self.table()
        data = pa.Table.from_pylist(rows, schema=self.schema())
        # merge_insert — единый батч «вставил новый / обновил существующий».
        # Старый построчный update уходил в O(n) прогодов при массовом перемещении
        # файлов; merge_insert делает то же за один вызов (lancedb >= 0.15).
        t.merge_insert("hash").when_matched_update_all().when_not_matched_insert_all().execute(data)
        return n_upd

    def prune_missing_hashes(self, keep_hashes: set) -> int:
        """Удаляет записи, чьего контента нет в keep_hashes.

        Перемещённые/переименованные файлы сохраняются (их хеш в сете),
        удаляются только реально удалённые файлы.
        """
        t = self.table(create=False)
        if t is None:
            return 0
        current = self.existing_hashes()
        to_delete = [h for h in current if h not in keep_hashes]
        for chunk in _chunks(sorted(to_delete), 500):
            quoted = ", ".join("'" + _check_hash(h).replace("'", "''") + "'"
                               for h in chunk)
            t.delete(f"hash IN ({quoted})")
        return len(to_delete)

    def delete_hashes(self, hashes: list) -> int:
        """Удаляет строки по списку хешей (каннами). Возвращает число реально
        удалённых записей (не чанков — иначе при некорректном хеше отчёт
        завышал бы количество)."""
        t = self.table(create=False)
        if t is None or not hashes:
            return 0
        current = self.existing_hashes()
        to_delete = [h for h in set(hashes) if h in current]
        for chunk in _chunks(sorted(to_delete), 500):
            quoted = ", ".join("'" + _check_hash(h).replace("'", "''") + "'"
                               for h in chunk)
            t.delete(f"hash IN ({quoted})")
        return len(to_delete)

    # ------------------------------------------------------------------- index
    def has_vector_index(self) -> bool:
        t = self.table(create=False)
        return bool(t.list_indices()) if t is not None else False

    def maybe_create_ann_index(self, log=print) -> bool:
        """IVF-PQ после INDEX_MIN_ROWS строк. Возвращает True, если создан."""
        t = self.table(create=False)
        if t is None:
            return False
        n = t.count_rows()
        if n < config.INDEX_MIN_ROWS or t.list_indices():
            return False
        num_partitions = max(1, int(round(math.sqrt(n))))
        num_sub_vectors = (self.dim // 16 if self.dim % 16 == 0
                           else max(1, self.dim // 8))
        log(f"Строю IVF-PQ индекс: {n} векторов, "
            f"партиций={num_partitions}, субвекторов={num_sub_vectors}...")
        try:  # новый unified API (lancedb >= 0.24)
            from lancedb.index import IvfPq
            t.create_index("vector", config=IvfPq(
                distance_type="cosine",
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors))
        except Exception:
            # legacy API для старых версий lancedb
            t.create_index(metric="cosine",
                           num_partitions=num_partitions,
                           num_sub_vectors=num_sub_vectors,
                           vector_column_name="vector",
                           replace=True)
        log("IVF-PQ индекс создан.")
        return True

    def optimize(self, log=print) -> None:
        t = self.table(create=False)
        if t is None:
            return
        try:
            t.optimize()
        except Exception as e:
            log(f"  (компактация пропущена: {e})")

    def all_rows(self, include_vector: bool = False,
                 columns: list | None = None) -> list:
        """Все строки для каталога. columns=None → все кроме vector (+vector).

        При поиске по позе/палитре передаём узкий список: не тащим face_vector
        (1536 float) и прочее, что в том режиме не читается."""
        t = self.table(create=False)
        if t is None or t.count_rows() == 0:
            return []
        if columns is None:
            columns = ["hash", "path", "mtime", "size", "thumb", "pose",
                       "face_vector", "palette", "content"]
            if include_vector:
                columns.append("vector")
        return t.to_arrow().select(columns).to_pylist()

    # ------------------------------------------------------------------ search
    def search(self, vector, k: int, column: str = "vector") -> list:
        """Топ-k ближайших по векторному столбцу (vector/face_vector)."""
        t = self.table(create=False)
        if t is None or t.count_rows() == 0:
            return []
        q = t.search(vector, vector_column_name=column).limit(k)
        if column == "vector" and self.has_vector_index():
            n_parts = max(1, int(round(math.sqrt(t.count_rows()))))
            nprobes = max(1, n_parts // config.ANN_NPROBES_DIVISOR)
            q = q.nprobes(nprobes).refine_factor(10)
        return q.to_list()

    def flat_cosine(self, query: list, column: str, k: int,
                    exclude: set | None = None) -> list:
        """Топ-k по косинусу над столбцом (плоский поиск, для pose/face)."""
        t = self.table(create=False)
        if t is None or t.count_rows() == 0:
            return []
        import numpy as np
        q = np.asarray(query, dtype=np.float32)
        nq = np.linalg.norm(q)
        if nq == 0:
            return []
        rows = t.to_arrow().select(["hash", "path", "thumb", column]).to_pylist()
        scored = []
        for r in rows:
            v = r.get(column)
            if not v:
                continue
            if exclude and r.get("hash") in exclude:
                continue
            nv = np.linalg.norm(v)
            if nv == 0:
                continue
            scored.append((float(np.dot(q, v) / (nq * nv)), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**r, "_score": s} for s, r in scored[:k]]

    # ------------------------------------------------------------------ info
    def db_size_bytes(self) -> int:
        total = 0
        for p in self.db_dir.rglob("*"):
            if p.is_file() and "thumbs" not in p.parts[len(self.db_dir.parts):]:
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def thumbs_info(self) -> tuple:
        count = 0
        size = 0
        if self.thumbs_dir.exists():
            for p in self.thumbs_dir.iterdir():
                if p.is_file():
                    count += 1
                    size += p.stat().st_size
        return count, size

"""Команда `index`: обход каталога, индексация с дедупликацией по хешу.

Дедупликация происходит на этапе добавления:
  * Хеш = sha1 содержимого — первичный ключ записи в LanceDB.
  * Если хеш уже есть в таблице (файл перемещён/переименован или не менялся) —
    вектор перевычислять не нужно, только обновляем путь/mtime/миниатюру.
  * Если хеш встретился повторно в этом же обходе — тоже пропускаем (дубль
    внутри одной папки).

Из-за дедупликации по хешу каждый файл читается целиком (хеш считается по
контенту, а не по имени/времени) — это цена корректной защиты от перемещения.
Полноценный skip по mtime отключён намеренно: при перемещении mtime/путь
меняются, и ориентироваться на них нельзя.
"""

import json
import time
from pathlib import Path

import numpy as np

from . import config
from .imaging import iter_image_files, make_thumbnail, prepare_image, sha1_bytes
from .model import DINOv2Embedder
from .pose import PoseDetector
from .store import ImageStore


def _batch_analyze(new_rows) -> tuple[list, list]:
    """Палитра (JSON цветов) + тип контента (JSON safe/nsfw) для пачки."""
    from . import analyze
    pals, contents = [], []
    for m in new_rows:
        pals.append(json.dumps(analyze.palette(m["img"])))
        contents.append(json.dumps(analyze.detect_content(m["img"])))
    return pals, contents


def _batch_people(new_rows, pose_detector, embedder, face_dim) -> tuple[list, list]:
    """Поза (JSON) + эмбеддинг лица для пачки изображений.

    Возвращает (pose_dicts, face_vecs): для каждого изображения — словарь позы
    или None, и список эмбеддинга лица (или нулевой вектор, если лица нет).
    """
    from .imaging import prepare_image
    pose_dicts, face_vecs = [], []
    face_imgs = []
    # Сначала вырезаем лица (лениво), потом эмбедим найденные.
    prepared = []
    for m in new_rows:
        pose = pose_detector.detect_pose(m["img"])
        pose_dicts.append(pose.to_dict() if pose is not None else {"kps": [], "score": 0.0})
        face = pose_detector.crop_face(m["img"])
        face_imgs.append(face)
    face_embs = embedder.embed_images([f for f in face_imgs if f]) if any(face_imgs) else []
    fi = iter(face_embs)
    for face in face_imgs:
        if face is not None:
            try:
                face_vecs.append(next(fi).tolist())
            except StopIteration:
                face_vecs.append([0.0] * face_dim)
        else:
            face_vecs.append([0.0] * face_dim)
    return pose_dicts, face_vecs


def run_index(dir_path: str, db_dir: str, model_dir: str | None = None,
              batch_size: int | None = None,
              recursive: bool = True, rebuild: bool = False, prune: bool = False,
              max_side: int = config.PREPROCESS_MAX_SIDE, log=print) -> dict:
    t0 = time.time()
    root = Path(dir_path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Каталог не найден: {root}")

    store = ImageStore(db_dir, config.MODEL)
    embedder = DINOv2Embedder(model_dir=model_dir, log=log)
    # Детектор поз/лиц — один на всю индексацию (ленивый, кэш модели).
    pose_detector = PoseDetector()
    face_dim = embedder.dim

    if batch_size is None:
        batch_size = config.default_batch_size(embedder.device)
    log(f"Каталог: {root} | рекурсивно: {'да' if recursive else 'нет'} | "
        f"батч: {batch_size} | сжатие перед векторизацией: до {max_side}px")
    if model_dir:
        log(f"Локальные веса модели: {Path(model_dir).resolve()}")

    if rebuild:
        store.drop()
        log("Таблица пересоздаётся (--rebuild).")

    files = list(iter_image_files(root, recursive=recursive,
                                  exclude_dirs=(store.db_dir.resolve(),)))
    if not files:
        raise SystemExit("Изображения не найдены (jpg/png/webp/bmp/gif/tiff).")
    log(f"Найдено файлов: {len(files)}")

    # Хеши, уже лежащие в таблице — для дедупликации и защиты от перемещения.
    existing_hashes = set() if rebuild else store.existing_hashes()
    log(f"Уже в индексе: {len(existing_hashes)} записей (по хешу).")

    # --- буферы для пакетного добавления ---------------------------------------
    new_rows: list = []        # новый контент -> нужен эмбеддинг
    upd_rows: list = []        # перемещённый контент -> обновить путь/mtime
    seen_hashes: set = set(existing_hashes)  # дедупликация внутри обхода
    added_hashes: set = set()  # хеши, добавленные в этот раз (тоже дедупликация)
    buf_len = 0                # размер буфера в памяти (изображения держим тут)
    stats = {"added": 0, "updated": 0, "failed": 0, "duplicates": 0}
    errors: list = []

    def flush() -> None:
        nonlocal buf_len
        if not new_rows and not upd_rows:
            buf_len = 0
            return
        if new_rows:
            embs = embedder.embed_with_retry(
                [m["img"] for m in new_rows], log=log)
            pose_dicts, face_vecs = _batch_people(new_rows, pose_detector,
                                                   embedder, face_dim)
            pals, contents = _batch_analyze(new_rows)
            vectorized = [{**m, "vector": emb.tolist(),
                           "pose": json.dumps(p),
                           "face_vector": fv,
                           "palette": pl, "content": c}
                          for m, emb, p, fv, pl, c in zip(
                              new_rows, embs, pose_dicts, face_vecs, pals, contents)]
            store.add_rows(vectorized)
            stats["added"] += len(vectorized)
        if upd_rows:
            stats["updated"] += store.upsert_rows(upd_rows)
        new_rows.clear()
        upd_rows.clear()
        buf_len = 0  # обнуляем: иначе buf_len >= threshold и flush лезет на каждый
        # следующий файл (построчный эмбеддинг вместо пачек — медленно).

    try:
        from tqdm import tqdm
    except ImportError:  # tqdm опционален
        tqdm = None
    progress = (tqdm(files, desc="Индексация", unit="img", ncols=100,
                     ascii=True) if tqdm else files)

    for f in progress:
        try:
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(added=stats["added"],
                                     updated=stats["updated"],
                                     dup=stats["duplicates"],
                                     err=stats["failed"])
            st = f.stat()
            if st.st_size > config.MAX_FILE_BYTES:
                stats["failed"] += 1
                errors.append((f, f"файл больше "
                               f"{config.MAX_FILE_BYTES // (1024*1024)} МБ"))
                continue

            # Хеш по содержимому: первичный ключ + защита от перемещения.
            data = f.read_bytes()
            h = sha1_bytes(data)

            if h in seen_hashes:
                # Дубль по контенту. Вектор не перевычисляем.
                stats["duplicates"] += 1
                if h in added_hashes:
                    continue  # тот же контент добавлен в этом обходе — пропуск
                if h in existing_hashes:
                    # Перемещён/переименован относительно прошлой индексации:
                    # обновляем путь/ftime/миниатюру, только если реально
                    # изменились путь или mtime (иначе перепись без пользы).
                    meta = store.get_meta(h)
                    if (not meta or meta.get("path") != str(f.resolve())
                            or meta.get("mtime") != st.st_mtime):
                        upd_rows.append({"hash": h,
                                         "path": str(f.resolve()),
                                         "mtime": st.st_mtime,
                                         "size": st.st_size,
                                         "thumb": str(make_thumbnail(
                                             data, store.thumbs_dir, h))})
                continue

            seen_hashes.add(h)
            added_hashes.add(h)
            new_rows.append({"hash": h,
                             "path": str(f.resolve()),
                             "mtime": st.st_mtime,
                             "size": st.st_size,
                             "thumb": str(make_thumbnail(
                                 data, store.thumbs_dir, h)),
                             "img": prepare_image(data, max_side=max_side)})
            buf_len += 1
            if buf_len >= config.STORE_FLUSH_ROWS:
                flush()
        except Exception as e:  # битые/нечитаемые файлы не ломают индексацию
            stats["failed"] += 1
            errors.append((f, repr(e)))

    flush()

    created_index = store.maybe_create_ann_index(log=log)

    removed = 0
    if prune:
        # Дедупликация по хешу: удаляем записи, чьего контента больше нет
        # среди файлов. Перемещённые/переименованные файлы сохраняются — их
        # хеш на месте, меняется только путь.
        current_hashes = {
            sha1_bytes(Path(f).read_bytes())
            for f in iter_image_files(root, recursive=recursive)}
        removed = store.prune_missing_hashes(current_hashes)

    total_rows = store.rows_count()
    if stats["added"] > 1000:
        store.optimize(log=log)

    dt = time.time() - t0
    speed = stats["added"] / dt if dt > 0 else 0.0
    log("")
    log(f"Готово за {dt:.1f} с ({speed:.1f} изобр/с)")
    log(f"Добавлено: {stats['added']} | Обновлено (перемещено/переименовано): "
        f"{stats['updated']} | Дублей пропущено: {stats['duplicates']} | "
        f"Ошибок: {stats['failed']}"
        + (f" | Удалено отсутствующих: {removed}" if prune else ""))
    log(f"Всего в таблице {store.table_name}: {total_rows} записей | "
        f"ANN-индекс: {'IVF-PQ' if created_index or store.has_vector_index() else 'плоский точный поиск'}")

    if errors:
        log(f"Последние ошибки (всего {len(errors)}):")
        for f, e in errors[:10]:
            log(f"  {f.name}: {e[:120]}")

    stats.update({"total_rows": total_rows, "elapsed": dt, "files": len(files),
                  "index_created": created_index})
    return stats

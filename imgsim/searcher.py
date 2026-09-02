"""Команда `search`: поиск похожих изображений и генерация HTML-галереи."""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from .imaging import make_thumbnail, prepare_image, sha1_bytes
from .model import DINOv2Embedder
from .pose import PoseDetector, Pose, cosine
from .store import ImageStore


def _search_pose(pose_vec, store, query_sha, top_k, min_score, log):
    """Поиск похожих поз: косинус эмбеддинга запроса над всеми позами."""
    if pose_vec is None:
        return []
    rows = store.all_rows()
    scored = []
    for r in rows:
        if r.get("hash") == query_sha:
            continue
        pose_json = r.get("pose")
        if not pose_json:
            continue
        try:
            d = json.loads(pose_json)
        except Exception:
            continue
        from .pose import Pose
        p = Pose.from_dict(d)
        s = cosine(pose_vec, p.vector())
        if s < min_score:
            continue
        scored.append({"path": r["path"], "score": s, "thumb": r["thumb"]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _search_palette(pvec, store, query_sha, top_k, min_score, log):
    """Поиск похожих по цветовой палитре: косинус вектора палитры запроса."""
    if not pvec:
        return []
    import json
    from . import analyze
    rows = store.all_rows()
    scored = []
    for r in rows:
        if r.get("hash") == query_sha:
            continue
        pal_json = r.get("palette")
        if not pal_json:
            continue
        try:
            pal = json.loads(pal_json)
        except Exception:
            continue
        v = analyze.palette_vector(pal)
        a = np.asarray(pvec, dtype=np.float32)
        b = np.asarray(v, dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue
        s = float(np.dot(a, b) / (na * nb))
        if s < min_score:
            continue
        scored.append({"path": r["path"], "score": s, "thumb": r["thumb"]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def run_search(query_image: str, db_dir: str, model_dir: str | None = None,
               top_k: int = 50, min_score: float = 0.0,
               out_path: str | None = None,
               mode: str = "image",
               max_side: int = config.PREPROCESS_MAX_SIDE, log=print) -> Path:
    t0 = time.time()
    query = Path(query_image).expanduser().resolve()
    if not query.is_file():
        raise SystemExit(f"Файл запроса не найден: {query}")

    store = ImageStore(db_dir, config.MODEL)
    total_rows = store.rows_count()
    if total_rows == 0:
        raise SystemExit(f"Индекс пуст. Сначала выполните:\n"
                         f"  imgsim index <каталог> --db {db_dir}")

    # DINOv2-модель и детектор поз/лиц нужны только для image/face/pose; в
    # режиме palette запрос — только палитра (PIL), тяжёлая модель не грузится.
    embedder = None
    pose_detector = None
    if mode != "palette":
        embedder = DINOv2Embedder(model_dir=model_dir, log=log)
        pose_detector = PoseDetector()

    # --- эмбеддинг/поза запроса ------------------------------------------------
    data = query.read_bytes()
    sha = sha1_bytes(data)
    thumb = make_thumbnail(data, store.thumbs_dir, sha)  # кэчируется по sha
    img = prepare_image(data, max_side=max_side)

    t_emb = time.time()
    if mode == "face":
        face = pose_detector.crop_face(img)
        emb = embedder.embed_images([face])[0] if face is not None else None
    elif mode == "pose":
        p = pose_detector.detect_pose(img)
        pose_vec = p.vector() if p is not None else None
    elif mode == "palette":
        from . import analyze
        pvec = analyze.palette_vector(analyze.palette(img))
    else:  # image
        emb = embedder.embed_images([img])[0]
    emb_ms = (time.time() - t_emb) * 1000

    # --- поиск -------------------------------------------------------------------
    t_srch = time.time()
    if mode == "pose":
        results = _search_pose(pose_vec, store, sha, top_k, min_score, log)
    elif mode == "palette":
        results = _search_palette(pvec, store, sha, top_k, min_score, log)
    else:
        col = "face_vector" if mode == "face" else "vector"
        if emb is None:
            log(f"В запросе не найдено {'лицо' if mode=='face' else 'изображение'}.")
            results = []
        else:
            raw = store.search(emb.tolist(), top_k + 10, column=col)
            results = []
            for r in raw:
                p = r["path"]
                if p == str(query):
                    continue
                # векторы нормированы (L2=1): cosine = 1 - dist²/2 (dist — L2)
                score = max(0.0, min(1.0, 1.0 - (float(r["_distance"]) ** 2) / 2.0))
                results.append({"path": p, "score": score, "thumb": r["thumb"]})
                if len(results) >= top_k:
                    break
    srch_ms = (time.time() - t_srch) * 1000

    # --- галерея -----------------------------------------------------------------
    from .gallery import render_gallery
    out = Path(out_path) if out_path else (
        config.results_dir() / f"search_{datetime.now():%Y%m%d_%H%M%S}.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    html = render_gallery(
        query_path=str(query),
        query_thumb=str(thumb),
        results=results,
        model_id=embedder.model_id if embedder else config.MODEL_ID,
        variant=embedder.variant if embedder else config.MODEL,
        dim=embedder.dim if embedder else config.MODEL_DIM,
        total_rows=total_rows,
        emb_ms=emb_ms,
        srch_ms=srch_ms,
        min_score=min_score,
        db_dir=str(store.db_dir.resolve()),
        mode=mode,
    )
    out.write_text(html, encoding="utf-8")

    dt = time.time() - t0
    log("")
    log(f"Запрос: {query} | режим: {mode}")
    log(f"Эмбеддинг: {emb_ms:.0f} мс | Поиск: {srch_ms:.0f} мс | "
        f"Всего: {dt * 1000:.0f} мс")
    log(f"Похожих найдено: {len(results)} из {total_rows} в индексе")
    for i, r in enumerate(results[:5], 1):
        log(f"  {i}. {r['score']:.3f}  {r['path']}")
    if len(results) > 5:
        log(f"  ... остальные в галерее")
    log(f"Галерея: {out.resolve()}")
    return out

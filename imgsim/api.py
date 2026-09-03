"""Локальный HTTP API + Vue-фронт для imgsim.

API (stdlib http.server — без новых зависимостей и build-шага):
  GET  /api/health        -> {ok: true}
  GET  /api/stats         -> статистика индекса
  GET  /api/rows?offset&limit -> каталог записей (hash/path/size/thumb)
  GET  /api/search?hash&top&min_score&mode -> похожие по хешу записи
  GET  /api/open?hash     -> исходный файл (с content-type)
  GET  /api/thumb?hash    -> миниатюра WebP
  POST /api/delete {hashes} -> удалённые записи
  GET  /                 -> Vue-фронт (single-file, Vue 3 через CDN)

Поиск по хешу не требует пере-эмбеддинга для image/face (берётся сохранённый
вектор записи); pose/palette пересчитываются из файла один раз.
"""

import json
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config
from .imaging import prepare_image
from .store import ImageStore, _check_hash
from .utils import distance_to_cosine


def _mime(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


# _api_open отдаёт только эти расширения — иначе /etc/passwd (is_file()
# истинен) утекал бы через поддельный путь в БД.
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".heic", ".tiff", ".raw"}


def _row_view(r: dict) -> dict:
    return {"hash": r.get("hash"), "path": r.get("path"),
            "thumb": r.get("thumb"), "size": r.get("size")}


def search_by_hash(store: ImageStore, h: str, top_k: int = 50,
                   min_score: float = 0.0, mode: str = "image") -> list[dict]:
    """Поиск похожих по хешу записи, режим image/face/pose/palette."""
    _check_hash(h)
    row = store.get_meta(h, with_vector=(mode == "image"))
    if row is None:
        return []

    if mode == "image":
        vec = row.get("vector")
        if not vec:
            return []
        raw = store.search(vec, top_k + 10, column="vector")
        out = []
        for r in raw:
            s = distance_to_cosine(r.get("_distance"))
            if s < min_score or r["path"] == row["path"]:
                continue
            out.append({**_row_view(r), "score": round(s, 4)})
        return out[:top_k]

    if mode == "face":
        vec = row.get("face_vector")
        if not vec:
            return []
        raw = store.search(vec, top_k + 10, column="face_vector")
        out = []
        for r in raw:
            s = distance_to_cosine(r.get("_distance"))
            if s < min_score or r["path"] == row["path"]:
                continue
            out.append({**_row_view(r), "score": round(s, 4)})
        return out[:top_k]

    # pose / palette: пересчёт вектора запроса из файла (один раз).
    if mode == "pose":
        from .pose import PoseDetector, cosine
        p = PoseDetector().detect_pose(prepare_image(Path(row["path"]).read_bytes()))
        pv = p.vector() if p is not None else None
        if not pv:
            return []
        from .searcher import _search_pose
        res = _search_pose(pv, store, h, top_k, min_score, lambda *a: None)
    elif mode == "palette":
        from . import analyze
        pvec = analyze.palette_vector(analyze.palette(prepare_image(
            Path(row["path"]).read_bytes())))
        from .searcher import _search_palette
        res = _search_palette(pvec, store, h, top_k, min_score, lambda *a: None)
    else:
        raise ValueError(f"неизвестный режим поиска: {mode}")
    # _search_pose/_search_palette отдают path/thumb/score — добавляем hash/size.
    by_path = {r["path"]: (r.get("hash"), r.get("size"))
               for r in store.all_rows(columns=["hash", "path", "size"])}
    out = []
    for r in res:
        hh, sz = by_path.get(r["path"], (None, None))
        out.append({"hash": hh, "path": r["path"], "thumb": r.get("thumb"),
                    "size": sz, "score": round(r["score"], 4)})
    return out


def _stats(store: ImageStore) -> dict:
    count, thumb_mb = store.thumbs_info()
    try:
        db_mb = store.db_size_bytes() / (1024 * 1024)
    except Exception:
        db_mb = 0.0
    return {"total_rows": store.rows_count(), "thumbs_count": count,
            "thumbs_mb": round(thumb_mb, 1), "db_mb": round(db_mb, 1),
            "has_ann_index": store.has_vector_index(), "dim": store.dim}


class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, db_dir=None, **kwargs):  # привязываем db_dir
        self._api_db_dir = db_dir
        super().__init__(*args, **kwargs)

    # -------------------------------------------------------------- helpers
    def _send(self, code: int, ctype: str, data: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, code: int, obj):
        self._send(code, "application/json; charset=utf-8",
                   json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _send_text(self, code: int, text: str):
        self._send(code, "text/plain; charset=utf-8", text.encode("utf-8"))

    def _send_file(self, path: Path):
        if not path.is_file():
            self._send_text(404, "Not found")
            return
        self._send(200, _mime(path), path.read_bytes())

    def _store(self) -> ImageStore:
        return ImageStore(self._api_db_dir, config.MODEL)

    def _read_json_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # -------------------------------------------------------------- routing
    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/api/health":
                return self._send_json(200, {"ok": True})
            if path == "/api/stats":
                return self._send_json(200, _stats(self._store()))
            if path == "/api/rows":
                return self._rows(qs)
            if path == "/api/search":
                return self._search(qs)
            if path == "/api/open":
                return self._original(qs)
            if path == "/api/thumb":
                return self._thumb(qs)
            if path in ("/", "/index.html"):
                return self._send_file(self._app_path())
            return self._send_text(404, "Not found")
        except (ValueError, KeyError) as e:
            return self._send_json(400, {"error": str(e)})
        except Exception as e:  # не ломаем соединение из-за одной записи
            return self._send_json(500, {"error": repr(e)})

    def do_POST(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        try:
            if path == "/api/delete":
                return self._delete()
            return self._send_text(404, "Not found")
        except (ValueError, KeyError) as e:
            return self._send_json(400, {"error": str(e)})
        except Exception as e:
            return self._send_json(500, {"error": repr(e)})

    # -------------------------------------------------------------- endpoints
    def _rows(self, qs: dict):
        offset = max(0, int(qs.get("offset", ["0"])[0]))
        limit = max(1, min(500, int(qs.get("limit", ["200"])[0])))
        store = self._store()
        rows = store.all_rows(columns=["hash", "path", "size", "thumb"])
        total = len(rows)
        page = rows[offset:offset + limit]
        self._send_json(200, {"total": total, "offset": offset,
                               "limit": limit, "items": [_row_view(r) for r in page]})

    def _search(self, qs: dict):
        h = qs.get("hash", [""])[0]
        if not h:
            raise ValueError("требуется ?hash=<sha1>")
        mode = qs.get("mode", ["image"])[0]
        if mode not in ("image", "face", "pose", "palette"):
            raise ValueError("mode должен быть image/face/pose/palette")
        top_k = max(1, int(qs.get("top", ["50"])[0]))
        min_score = float(qs.get("min_score", ["0.0"])[0])
        results = search_by_hash(self._store(), h, top_k=top_k,
                                 min_score=min_score, mode=mode)
        self._send_json(200, {"mode": mode, "count": len(results), "items": results})

    def _original(self, qs: dict):
        h = qs.get("hash", [""])[0]
        _check_hash(h)
        meta = self._store().get_meta(h)
        if not meta:
            return self._send_text(404, "Not found")
        # resolve + whitelist расширений: /api/open отдаёт только реальные
        # изображения. Блокирует path traversal через поддельный путь в БД
        # (например /etc/passwd — он is_file(), но не картинка).
        p = Path(meta["path"]).resolve()
        if p.suffix.lower() not in _IMAGE_EXT or not p.is_file():
            return self._send_text(404, "Not found")
        self._send_file(p)

    def _thumb(self, qs: dict):
        h = qs.get("hash", [""])[0]
        _check_hash(h)
        meta = self._store().get_meta(h)
        if not meta or not meta.get("thumb"):
            return self._send_text(404, "Not found")
        self._send_file(Path(meta["thumb"]))

    def _delete(self):
        body = self._read_json_body()
        hashes = body.get("hashes") if isinstance(body, dict) else None
        if not isinstance(hashes, list) or not hashes:
            raise ValueError("нужно {\"hashes\": [...]}")
        for h in hashes:
            _check_hash(h)  # валидация до любого обращения к БЗ
        deleted = self._store().delete_hashes(list(hashes))
        self._send_json(200, {"deleted": deleted})

    # -------------------------------------------------------------- static
    def _app_path(self) -> Path:
        return Path(__file__).parent / "templates" / "app.html"

    def log_message(self, *args):  # тише, без спам-лога каждого запроса
        pass


def make_handler(db_dir: str):
    """Фабрика handler'а с привязанным db_dir (http.server не умеет аргументы)."""
    return lambda *a, **k: APIHandler(*a, db_dir=db_dir, **k)


def run(*, db_dir: str, host: str = "127.0.0.1", port: int = 8137):
    store = ImageStore(db_dir, config.MODEL)
    if not store.exists():
        raise SystemExit(f"Индекс не найден: {store.db_dir.resolve()}. "
                         f"Сначала: imgsim index <каталог> --db {db_dir}")
    srv = ThreadingHTTPServer((host, port), make_handler(db_dir))
    print(f"imgsim API + фронт: http://{host}:{port}/")
    print(f"  База: {store.db_dir.resolve()}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        srv.server_close()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="HTTP API + Vue-фронт для просмотра/поиска/удаления в инксе")
    ap.add_argument("--db", default=config.DEFAULT_DB_DIR, help="каталог базы")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8137)
    args = ap.parse_args(argv)
    run(db_dir=args.db, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

"""Локальный сервер для browse.html.

Отдаёт HTML-страницу и открывает оригиналы изображений по http://, а не
file:// (браузеры блокируют file:// над http://). Кнопка «Открыть» и клик по
изображению запрашивают /open?hash=<hash>; сервер ищет путь записи в базе
по хешу (первичный ключ) и отдаёт исходный файл с правильным content-type.

    python3 -m imgsim serve --db ./image_db --dir . --port 8137
"""

import argparse
import mimetypes
import posixpath
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config
from .store import ImageStore


def _guess_mime(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def _serve(db_dir: str, root: Path, host: str, port: int) -> None:
    root = root.resolve()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802 (API-имя http.server)
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/open":
                self._serve_original(urllib.parse.parse_qs(parsed.query))
                return
            self._serve_static(urllib.parse.unquote(parsed.path) or "/")

        def _serve_original(self, qs: dict) -> None:
            h = qs.get("hash", [""])[0]
            try:
                meta = ImageStore(db_dir, config.MODEL).get_meta(h)
                p = Path(meta["path"])
            except Exception:
                p = None
            if p and p.is_file():
                data = p.read_bytes()
                self._respond(_guess_mime(p), data)
            else:
                self._respond_text(404, "Not found")

        def _serve_static(self, urlpath: str) -> None:
            rel = posixpath.normpath(urlpath.lstrip("/"))
            if rel.startswith(".."):
                self._respond_text(403, "Forbidden")
                return
            target = (root / rel)
            if target.is_dir():
                target = target / "index.html"
            if target.is_file():
                try:
                    data = target.read_bytes()
                except OSError:
                    self._respond_text(404, "Not found")
                    return
                self._respond(_guess_mime(target), data)
            else:
                self._respond_text(404, "Not found")

        def _respond(self, ctype: str, data: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _respond_text(self, code: int, text: str) -> None:
            data = text.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # тише, без спам-лога каждого запроса
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Индекс: http://{host}:{port}/   (остановить: Ctrl+C)")
    print(f"  HTML:  {root}")
    print(f"  База:  {db_dir}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        srv.server_close()


def run(*, db_dir: str, root: str, host: str = "127.0.0.1", port: int = 8137):
    root_path = Path(root)
    if not root_path.exists():
        raise SystemExit(f"Каталог с HTML не найден: {root_path.resolve()}")
    _serve(db_dir, root_path, host, port)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Отдать browse.html + открыть оригиналы по http")
    ap.add_argument("--db", default=config.DEFAULT_DB_DIR, help="каталог базы")
    ap.add_argument("--dir", default=str(Path.cwd()), help="каталог с HTML (по умолчанию текущий)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8137)
    args = ap.parse_args(argv)
    run(db_dir=args.db, root=args.dir, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

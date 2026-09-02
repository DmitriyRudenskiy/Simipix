"""CLI imgsim: index / search / stats."""

import argparse
import sys
import time
from pathlib import Path

from . import __version__, config
from .model import pick_device


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=config.DEFAULT_DB_DIR,
                   help=f"каталог базы LanceDB (по умолчанию "
                        f"{config.DEFAULT_DB_DIR})")
    p.add_argument("--model-dir", default=None,
                   help=f"локальные веса модели в {config.MODELS_DIR} "
                        f"(по умолчанию — с HuggingFace Hub)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="imgsim",
        description="Индексация изображений (DINOv2-giant) и поиск похожих "
                    "(LanceDB). Всегда используется самая большая модель.")
    ap.add_argument("--version", action="version",
                    version=f"imgsim {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="проиндексировать каталог изображений")
    p_idx.add_argument("dir", help="каталог с изображениями")
    _add_common(p_idx)
    p_idx.add_argument("--batch", type=int, default=None,
                       help="размер батча (по умолчанию: 16 на GPU, 4 на CPU)")
    p_idx.add_argument("--no-recursive", action="store_true",
                       help="не заходить в подкаталоги")
    p_idx.add_argument("--rebuild", action="store_true",
                       help="пересоздать таблицу с нуля")
    p_idx.add_argument("--prune", action="store_true",
                       help="удалить из индекса записи о несуществующих файлах")
    p_idx.add_argument("--max-side", type=int,
                       default=config.PREPROCESS_MAX_SIDE,
                       help="сжатие перед векторизацией, сторона px "
                            f"(по умолчанию {config.PREPROCESS_MAX_SIDE})")

    p_srch = sub.add_parser("search",
                            help="найти похожие на изображение-запрос")
    p_srch.add_argument("image", help="путь к изображению-запросу")
    _add_common(p_srch)
    p_srch.add_argument("--top", type=int, default=50,
                        help="сколько похожих показать (по умолчанию 50)")
    p_srch.add_argument("--min-score", type=float, default=0.0,
                        help="начальный порог схожести в галерее, 0..1")
    p_srch.add_argument("--out", default=None,
                        help="путь к HTML-галерее (по умолчанию "
                             "results/search_<время>.html)")
    p_srch.add_argument("--max-side", type=int,
                        default=config.PREPROCESS_MAX_SIDE,
                        help="сжатие запроса перед векторизацией, px")
    p_srch.add_argument("--mode", choices=["image", "face", "pose", "palette"],
                        default="image",
                        help="режим поиска: image — похожие изображения, "
                             "face — похожие люди, pose — одинаковые позы, "
                             "palette — похожие по цветовой палитре")

    p_st = sub.add_parser("stats", help="статистика индекса")
    _add_common(p_st)

    p_browse = sub.add_parser(
        "browse", help="открыть страницу с каталогом индекса и поиском")
    _add_common(p_browse)
    p_browse.add_argument("--top", type=int, default=10,
                          help="сколько похожих показать на карточке "
                               "(по умолчанию 10)")
    p_browse.add_argument("--out", default=None,
                          help="путь к HTML-странице (по умолчанию "
                               "results/browse_<время>.html)")

    p_dup = sub.add_parser(
        "find_duplicates",
        help="найти визуальные дубликаты и сгенерировать HTML для удаления")
    _add_common(p_dup)
    p_dup.add_argument("--threshold", type=float, default=0.85,
                       help="порог cosine-схожести для дубля, 0..1 "
                            "(по умолчанию 0.85)")
    p_dup.add_argument("--out", default=None,
                       help="путь к HTML-отчёту (по умолчанию "
                            "results/duplicates_<время>.html)")

    p_del = sub.add_parser(
        "delete", help="удалить записи из индекса по хешу (только по команде)")
    _add_common(p_del)
    p_del.add_argument("hashes", nargs="+", help="хеши записей к удалению")

    p_srv = sub.add_parser(
        "serve", help="отдать browse.html и открыть оригиналы по http (не file://)")
    p_srv.add_argument("--dir", default=str(Path.cwd()),
                       help="каталог с HTML-страницей (по умолчанию текущий)")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8137)
    _add_common(p_srv)

    p_web = sub.add_parser(
        "web", help="HTTP API + Vue-фронт для просмотра/поиска/удаления в инксе")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8137)
    _add_common(p_web)

    return ap


def _cmd_index(args) -> int:
    import signal
    from .indexer import run_index
    from .lock import acquire, release
    acquire(args.db)  # замок на каталог базы на время индексации

    def _release(signum, frame):
        release(args.db)  # на kill (SIGTERM)/Ctrl-C finally не срабатывает
        raise KeyboardInterrupt
    old_term = signal.signal(signal.SIGTERM, _release)
    old_int = signal.signal(signal.SIGINT, _release)
    try:
        run_index(args.dir, args.db, model_dir=args.model_dir, batch_size=args.batch,
                  recursive=not args.no_recursive, rebuild=args.rebuild,
                  prune=args.prune, max_side=args.max_side)
    finally:
        release(args.db)
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    return 0


def _cmd_search(args) -> int:
    if not 0.0 <= args.min_score <= 1.0:
        print("--min-score должен быть в диапазоне 0..1", file=sys.stderr)
        return 2
    from .searcher import run_search
    run_search(args.image, args.db, model_dir=args.model_dir, top_k=args.top,
               min_score=args.min_score, out_path=args.out,
               mode=args.mode, max_side=args.max_side)
    return 0


def _cmd_browse(args) -> int:
    from .browse import run_browse
    from .store import ImageStore
    store = ImageStore(args.db, config.MODEL)
    if not store.exists():
        raise SystemExit(f"Индекс не найден: {store.db_dir.resolve()}")
    total = store.rows_count()
    embed = total <= config.BROWSE_MAX_EMBED
    rows = store.all_rows(include_vector=embed)
    run_browse(store=store, rows=rows, embed=embed, top_k=args.top,
               model_id=config.MODEL_ID, dim=config.MODEL_DIM,
               total_rows=total, out_path=args.out)
    return 0


def _cmd_find_duplicates(args) -> int:
    from .find_duplicates import run_find_duplicates
    if not 0.0 < args.threshold < 1.0:
        print("--threshold должен быть в диапазоне 0..1", file=sys.stderr)
        return 2
    run_find_duplicates(args.db, model_dir=args.model_dir,
                        threshold=args.threshold, out_path=args.out)
    return 0


def _cmd_serve(args) -> int:
    from .serve import run
    run(db_dir=args.db, root=args.dir, host=args.host, port=args.port)
    return 0


def _cmd_web(args) -> int:
    from .api import run
    run(db_dir=args.db, host=args.host, port=args.port)
    return 0


def _cmd_delete(args) -> int:
    from .store import ImageStore
    store = ImageStore(args.db, config.MODEL)
    if not store.exists():
        raise SystemExit(f"Индекс не найден: {store.db_dir.resolve()}")
    existing = store.existing_hashes()
    want = set(args.hashes)
    missing = want - existing
    if missing:
        print(f"В индексе нет записей с хешами: {', '.join(sorted(missing)[:5])}"
              + ("…" if len(missing) > 5 else ""), file=sys.stderr)
    to_delete = want & existing
    if not to_delete:
        print("Нечего удалять: выбранные хеши отсутствуют в индексе.")
        return 1
    n = store.delete_hashes(to_delete)
    log_removed = ", ".join(sorted(to_delete)[:10]) + ("…" if len(to_delete) > 10 else "")
    print(f"Удалено из индекса: {n} записей ({log_removed})")
    print("После удаления файлов пересобрать индекс: imgsim index <каталог> "
          f"--db {args.db} --prune")
    return 0


def _cmd_stats(args) -> int:
    import lancedb
    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"База не найдена: {db_path.resolve()}")
        return 0
    db = lancedb.connect(str(db_path))
    names = db.table_names()
    if not names:
        print(f"База пуста: {db_path.resolve()}")
        return 0

    dev = pick_device()
    print(f"База: {db_path.resolve()} | модель: {config.MODEL_ID} | "
          f"устройство инференса: {dev}")
    for name in sorted(names):
        t = db.open_table(name)
        rows = t.count_rows()
        try:
            dim = t.schema.field("vector").type.list_size
        except Exception:
            dim = config.MODEL_DIM
        try:
            idx = t.list_indices()
        except Exception:
            idx = []
        idx_desc = "нет (плоский точный поиск)"
        if idx:
            det = ""
            try:
                det = (f"партиций~{int(rows ** 0.5)}, "
                       f"индексировано {idx[0].num_indexed_rows}")
            except Exception:
                pass
            idx_desc = f"IVF-PQ {det}"
        print(f"\n[{name}]")
        print(f"  Записей:       {rows}")
        print(f"  Размерность:   {dim}")
        print(f"  ANN-индекс:    {idx_desc}")
    # миниатюры
    thumbs = db_path / "thumbs"
    if thumbs.exists():
        files = [p for p in thumbs.iterdir() if p.is_file()]
        size_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
        print(f"\nМиниатюры: {len(files)} файлов, {size_mb:.1f} МБ "
              f"({thumbs})")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from .lock import is_locking
    if (args.cmd in ("index", "search", "browse", "delete", "serve")
            and is_locking(args.db)):
        print(f"Индекс занят — идёт индексация (db={args.db}). Повторные "
              f"index/search/browse/delete дождитесь завершения.")
        return 1
    t0 = time.time()
    try:
        rc = {"index": _cmd_index, "search": _cmd_search,
              "stats": _cmd_stats, "browse": _cmd_browse,
              "find_duplicates": _cmd_find_duplicates, "delete": _cmd_delete,
              "serve": _cmd_serve, "web": _cmd_web}
        rc = rc[args.cmd](args)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130
    except SystemExit as e:
        # ошибки с понятным сообщением поднимаются как SystemExit(str)
        if e.code and isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 1
        raise
    if args.cmd in ("index", "search"):
        print(f"Время выполнения: {time.time() - t0:.1f} с")
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())

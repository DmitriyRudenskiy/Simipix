"""Команда `find_duplicates`: поиск визуальных дубликатов в индексе.

Использует уже сохранённые векторы DINOv2 (L2-нормализованные): косинус =
скалярное произведение, поэтому вектора не перевычисляются. Группирует дубликаты
(связные компоненты по порогу cosine) и генерирует HTML-отчёт как в
VisionEmbed-CLI/find_duplicates.py: группы с миниатюрами, чекбоксами и кнопкой
«Удалить отмеченное» — генерирует rm-команды и команды удаления из индекса.

Удаление только по решению пользователя: команды генерируются, а не выполняются.

Хеши: дубликаты по ОДНОМУ содержимому во время индексации сливаются в одну
запись (защита от перемещения/переименования), поэтому здесь ищутся ВИЗУАЛЬНЫЕ
дубликаты — разные файлы, похожие картинки.
"""

import html as html_mod
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config


def _fmt_size(n: int) -> str:
    v = float(n)
    for unit in ("B", "КБ", "МБ", "ГБ"):
        if v < 1024 or unit == "ГБ":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство двух векторов (L2-нормализованным ≈ скалярное произведение)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    d = float(np.dot(a, b) / (na * nb + 1e-9))
    return max(0.0, min(1.0, d))


def _find_groups(rows: list, threshold: float):
    """Связные компоненты по cosine >= threshold (union-find).

    Возвращает (groups, scores): groups — список списков индексов записей,
    scores[i] — cosine записи i относительно образца группы (крупнейшего файла).

    ponytail: O(n) памяти на матрицу сходства, поиск пар — векторизованно
    (np.triu). Для больших каталогов (десятки тысяч) добавлять поиск
    кандидатов через ANN-индекс, а не полную матрицу.
    """
    n = len(rows)
    vectors = np.array([r["vector"] for r in rows], dtype=np.float32)
    sim = vectors @ vectors.T  # cosine, т.к. vectors уже L2-нормализованы

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = np.nonzero(np.triu(sim, k=1) >= threshold)
    for a, b in zip(pairs[0].tolist(), pairs[1].tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)

    groups, scores = [], []
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        sample = max(idxs, key=lambda i: rows[i].get("size", 0))
        sv = vectors[sample]
        groups.append(sorted(idxs))
        scores.append({i: _cosine(sv, vectors[i]) for i in idxs})
    return groups, scores


_CSS = """
:root{--bg:#0f1115;--panel:#171a21;--card:#1c2029;--line:#262b36;--text:#e8eaf0;
--muted:#8b93a5;--accent:#5b9dff;--keep:#3fb970;--del:#e35d5d}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,'Segoe UI',
Roboto,'Noto Sans',Arial,sans-serif;padding:28px clamp(16px,4vw,48px) 70px}
header{max-width:1240px;margin:0 auto 22px}
h1{font-size:22px;font-weight:700}
.meta{color:var(--muted);margin-top:8px}
.meta b{color:var(--text)}
.grid{max-width:1240px;margin:0 auto;display:grid;gap:18px}
.group{background:var(--panel);border:1px solid var(--line);border-radius:12px;
overflow:hidden}
.gh{display:flex;align-items:center;gap:12px;padding:12px 16px;
border-bottom:1px solid var(--line);background:#13161c}
.gh .cnt{color:var(--muted);font-weight:500}
.gh .avg{margin-left:auto;color:var(--accent);font-weight:700}
.gh .avg b{color:var(--del)}
.cards{display:flex;flex-wrap:wrap;gap:12px;padding:16px}
.card{width:186px;background:var(--card);border:1px solid var(--line);
border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.card.keep{border-color:var(--keep)}
.card .thumb{aspect-ratio:1/1;object-fit:cover;width:100%;display:block}
.card .body{padding:9px 11px;display:flex;flex-direction:column;gap:6px;flex:1}
.card .nm{font-size:12.5px;word-break:break-all;line-height:1.35}
.card .row{display:flex;justify-content:space-between;align-items:center;
font-size:11px;color:var(--muted)}
.card .sc{color:var(--keep);font-weight:700}
.card .ck{margin-top:auto;display:flex;align-items:center;gap:7px;font-size:12px;
color:var(--muted);cursor:pointer;padding-top:4px}
.card input{width:15px;height:15px;accent-color:var(--del);cursor:pointer}
.sticky{position:fixed;bottom:0;left:0;right:0;background:#13161c;border-top:1px
solid var(--line);padding:12px 20px;display:flex;gap:14px;align-items:center;
justify-content:center;z-index:20}
button{background:var(--accent);color:#fff;border:none;border-radius:8px;
padding:11px 22px;font-size:14px;font-weight:700;cursor:pointer}
button:hover{filter:brightness(1.08)}
.hint{color:var(--muted);font-size:13px}
.empty{max-width:1240px;margin:40px auto;color:var(--muted);text-align:center;
font-size:15px;padding:40px;border:1px dashed var(--line);border-radius:12px}
"""


_JS = """
function gen(){
  var hashes=[].map.call(document.querySelectorAll('.dchk:checked'),
    function(c){return c.getAttribute('data-hash');});
  var paths=[].map.call(document.querySelectorAll('.dchk:checked'),
    function(c){return c.getAttribute('data-path');});
  var db=__DB_DIR__;
  if(!hashes.length){alert('Ничего не выбрано для удаления.');return;}
  var L=['#!/bin/bash','set -e','','# 1) Удалить из индекса (только выбранные записи):'];
  L.push('imgsim delete '+hashes.map(function(h){return \"'\"+h.replace(/'/g,\"'\\\\''\")+\"'\";})
    .join(' ')+' --db '+db);
  L.push('','rm');
  paths.forEach(function(p){L.push(\"  '\"+p.replace(/'/g,\"'\\\\''\")+\"'\");});
  L.push('# 3) Пересобрать индекс с --prune для чистоты:');
  L.push('#   imgsim index <каталог> --db '+db+' --prune');
  document.getElementById('out').value=L.join('\\n').select();
}
"""


def _uri(p: str) -> str:
    return "file:///" + p.replace(os.sep, "/") if os.name == "nt" else "file://" + p


def _card_html(r, h, p, keep, sc_i) -> str:
    name = Path(p).name
    esc = html_mod.escape
    card = '<div class="card' + (' keep' if keep else '') + ' data-hash="' + esc(h) + '">'
    card += '<img class="thumb" src="' + esc(_uri(p)) + '" alt="' + esc(name) + '"'
    card += ' onerror="this.style.opacity=.3;this.alt=\'нет файла\'">'
    card += '<div class="body">'
    card += '<div class="nm">' + esc(name) + '</div>'
    card += '<div class="row"><span>' + _fmt_size(r.get("size", 0)) + \
            '</span><span class="sc">' + format(sc_i, '.2f') + '</span></div>'
    label = 'образец (сохранить)' if keep else 'удалить'
    card += '<label class="ck"><input type="checkbox" class="dchk" '
    card += 'data-hash="' + esc(h) + '" data-path="' + esc(p) + '">' + esc(label) + '</label>'
    card += '</div></div>'
    return card


def render_html(groups: list[list[int]], scores: list[dict], rows: list,
                threshold: float, model_id: str, dim: int,
                total_rows: int, db_dir: str) -> str:
    parts = [
        '<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">',
        '<title>' + str(len(groups)) + ' групп дубликатов</title><style>' + _CSS +
        '</style></head><body><header><h1>Дубликаты изображений</h1>',
        '<div class="meta">Модель: <b>' + html_mod.escape(model_id) + '</b> (' +
        str(dim) + '-dim) · порог cosine: <b>' + format(threshold, '.2f') +
        '</b> · записей в индексе: <b>' + str(total_rows) +
        '</b> · найдено групп дубликатов: <b>' + str(len(groups)) + '</b></div></header>',
    ]

    if not groups:
        parts.append('<div class="empty">Дубликатов выше порога ' +
                     format(threshold, '.2f') + ' не найдено. Все изображения '
                     'уникальны.</div>')
    else:
        parts.append('<div class="grid">')
        for gi, g in enumerate(groups, 1):
            sc = scores[gi - 1]
            sample = g[0]
            avg = sum(sc[i] for i in g if i != sample) / (len(g) - 1)
            parts.append('<div class="group"><div class="gh">')
            parts.append('<span class="tag">Группа #' + str(gi) + '</span>')
            parts.append('<span class="cnt">' + str(len(g)) + ' файлов</span>')
            parts.append('<span class="avg">средняя схожесть: <b>' +
                         format(avg, '.2f') + '</b></span></div><div class="cards">')
            for i in g:
                card = _card_html(rows[i], rows[i]["hash"], rows[i]["path"],
                                  i == sample, sc[i])
                parts.append(card)
            parts.append('</div></div>')
        parts.append('</div>')

    parts.append('<div class="sticky"><button onclick="gen()">Удалить отмеченное'
                 '</button><span class="hint">Генерирует `imgsim delete` + rm — '
                 ' только после проверки. <textarea id="out" style="position:'
                 'absolute;left:-99999px"></textarea></span></div><script>' +
                 _JS.replace('__DB_DIR__', repr(db_dir)) + '</script></body></html>')
    return ''.join(parts)


def run_find_duplicates(db_dir: str, model_dir: str | None = None,
                        threshold: float = 0.85, out_path: str | None = None,
                        log=print) -> Path:
    t0 = time.time()
    from .store import ImageStore
    store = ImageStore(db_dir, config.MODEL)
    total_rows = store.rows_count()
    if total_rows == 0:
        raise SystemExit(f'Индекс пуст. Сначала:\n  imgsim index <каталог> --db {db_dir}')

    rows = store.all_rows(include_vector=True)
    if len(rows) < 2:
        log('В индексе меньше двух записей — дубликатов не бывает.')
        return Path(out_path or '(не сгенерировано)')

    groups, scores = _find_groups(rows, threshold)

    out = Path(out_path) if out_path else (
        config.results_dir() / f'duplicates_{datetime.now():%Y%m%d_%H%M%S}.html')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_html(groups, scores, rows, threshold, config.MODEL, config.MODEL_DIM,
                    total_rows, str(store.db_dir.resolve())),
        encoding='utf-8')

    log(f'Записей в инксе: {total_rows} | групп дубликатов: {len(groups)} '
        f'(cosine ≥ {threshold:.2f})')

    # Стабильная копия «последнего отчёта» — рабочая ссылка из browse
    # (results/duplicates.html) независимо от времени в имени файла.
    try:
        (config.results_dir() / "duplicates.html").write_text(
            out.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        pass
    for gi, g in enumerate(groups[:20], 1):
        names = [Path(rows[i]['path']).name for i in g]
        log(f'  {gi}. {' '.join(names)}')
    if len(groups) > 20:
        log(f'  ... ещё {len(groups) - 20} групп')
    log(f'HTML: {out.resolve()} ({time.time() - t0:.1f} с)')
    return out

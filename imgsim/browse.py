"""Страница просмотра индекса: галерея всех записей с фильтрами и поиском.

Генерирует самодостаточный HTML-файл — открывается на любой машине без доступа
к базе. Возможности (всё на клиенте, без модели):
  1. Текстовый поиск по имени/путю — мгновенно.
  2. Фильтр по группе (папка-категория в пути).
  3. Постраничная навигация сверху и снизу.
  4. «Похожие» (cosine по вектору модели) — на малых каталогах; вектора
     встроены в HTML, поэтому вычисляется в браузере.
  5. Под каждым фото: палитра (до 5 домантных цветов) и поза (топ-5 ключевых
     точек DW-Pose) квадратиками.

Кнопка «Открыть» открывает оригинал через /open?hash= (прокси сервера
imgsim serve, а не file://, которое браузеры блокируют над http://). Ссылка
«Дубликаты» открывает HTML, сгенерированный `find_duplicates`.
"""

import base64
import html as html_mod
import json
import time
from datetime import datetime
from pathlib import Path

from . import config

_MIME = {"jpg": "jpeg", "jpeg": "jpeg", "webp": "webp", "png": "png",
         "gif": "gif", "bmp": "bmp"}

_PER_PAGE_DEFAULT = 24


def _meta(path: str) -> tuple[str, str, list[str]]:
    """Описание / группа-категория / теги из структуры пути.
    Группа — родительская папка; теги — 2 папки выше неё."""
    p = Path(path)
    dirs = [d for d in p.parts[:-1] if d and d not in ("/", ".", p.name)]
    if not dirs:
        return p.stem, p.parent.name, []
    category = dirs[-1]
    tags = dirs[-3:-1] if len(dirs) > 1 else []
    return p.stem, category, tags


def _json_field(r: dict, key: str) -> list | dict:
    """Распарсить JSON-поле записи (в БД хранится строкой)."""
    v = r.get(key)
    if not v:
        return {}
    try:
        return json.loads(v) if isinstance(v, str) else v
    except Exception:
        return {}


def _content_badge(r: dict) -> str:
    """Бейдж типа контента (safe/nsfw/unknown), если контент проанализирован."""
    c = _json_field(r, "content")
    if not isinstance(c, dict) or not c.get("label"):
        return ""
    label = c["label"]
    cls = "nsfw" if label == "nsfw" else ("safe" if label == "safe" else "unknown")
    ru = {"safe": "безопасно", "nsfw": "NSFW", "unknown": "—"}.get(label, "?")
    return (f'<span class="content {cls}" title="контент: {html_mod.escape(str(label))}">'
            f'{html_mod.escape(ru)}</span>')


def _palette_squares(r: dict, n: int = 5) -> str:
    """Полоска до N домантных цветов (визуальная палитра)."""
    pal = _json_field(r, "palette")
    if not isinstance(pal, list) or not pal:
        return ""
    items = [it for it in pal[:n] if it.get("hex")]
    if not items:
        return ""
    chips = ""
    for it in items:
        hexc = html_mod.escape(str(it.get("hex", "")))
        pct = html_mod.escape(f"{it.get('percent', 0):.0f}%")
        chips += (f'<span class="sw" style="background:{hexc}" '
                  f'title="{hexc} · {pct}" />')
    return (f'<div class="strip strip-pal" title="палитра (топ-{len(items)})">{chips}</div>')


def _pose_squares(r: dict, n: int = 5) -> str:
    """Полоска топ-N ключевых точек позы (DW-Pose) как квадратики в мини-сетке."""
    d = _json_field(r, "pose")
    if not isinstance(d, dict):
        return ""
    pts = []
    for pair in (d.get("kps") or []):
        try:
            x, y, c = float(pair[0]), float(pair[1]), float(pair[2])
        except (TypeError, ValueError, IndexError):
            continue
        if not (x == x and y == y and c == c):  # nan-пропуск
            continue
        pts.append((x, y, c))
    if not pts:
        return ""
    pts.sort(key=lambda t: t[2], reverse=True)
    top = pts[:n]
    size, pad, s = 66, 7, 7

    def col(c: float) -> str:
        if c >= 0.6:
            return "#2e9e5b"
        if c >= 0.35:
            return "#c9962a"
        return "#b0b7c2"

    rects = "".join(
        f'<rect x="{x * (size - 2 * pad) + pad - s / 2:.1f}" '
        f'y="{y * (size - 2 * pad) + pad - s / 2:.1f}" width="{s}" height="{s}" '
        f'rx="1.5" fill="{col(c)}" opacity="0.9"/>'
        for (x, y, c) in top)
    svg = (f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
           f'preserveAspectRatio="xMidYMid meet" '
           f'title="поза (DW-Pose): топ-{len(top)} точек">')
    svg += (f'<rect x="0" y="0" width="{size}" height="{size}" rx="6" '
            f'fill="#f2f3f5" stroke="#e0e3e8"/>' + rects + "</svg>")
    return (f'<div class="strip strip-pose" title="поза (DW-Pose)">{svg}</div>')


def _fmt_size(n: int) -> str:
    v = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if v < 1024 or unit == "ГБ":
            return f"{v:.0f} {unit}" if unit == "Б" else f"{v:.1f} {unit}"
        v /= 1024


def _data_uri(path) -> str:
    """Встраивает миниатюру в HTML как data URI (страница самодостаточная)."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        raw = base64.b64decode(
            "UklGRiYAAABXRUJQVlA4IBoAAAAwAQCdASoCAAIADMDOJZwAA3AA/ubNY4AAAA==")  # 2x2 webp
    mime = _MIME.get(Path(path).suffix.lstrip(".").lower(), "png")
    return f"data:image/{mime};base64,{base64.b64encode(raw).decode('ascii')}"


_CSS = """
:root{--bg:#eef0f4;--panel:#fff;--card:#fff;--line:#dde1e8;--text:#1e2430;
--muted:#6b7483;--accent:#3564d6;--good:#2e9e5b;--warn:#c9962a;--shadow:0 1px 2px
rgba(24,34,60,.06),0 8px 24px rgba(24,34,60,.07)}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);
font:14px/1.48 -apple-system,'Segoe UI',Roboto,'Noto Sans',Arial,sans-serif;
padding:0 0 64px}
.wrap{max-width:1440px;margin:0 auto;padding:0 clamp(16px,4vw,48px)}
header{padding:26px 0 18px}
h1{font-size:23px;font-weight:700;letter-spacing:.2px}
.sub{color:var(--muted);margin-top:7px;display:flex;flex-wrap:wrap;gap:8px 18px}
.sub b{color:var(--text);font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:13px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:20px;
padding:4px 12px;font-size:12.5px;color:var(--muted)}
.chip b{color:var(--accent)}
.chip a{color:var(--accent);text-decoration:none;font-weight:600}
.chip a:hover{text-decoration:underline}
.controls{position:sticky;top:10px;z-index:5;display:flex;flex-wrap:wrap;
gap:12px;align-items:center;background:rgba(255,255,255,.92);
border:1px solid var(--line);border-radius:14px;padding:11px 15px;
margin-bottom:16px;box-shadow:var(--shadow);backdrop-filter:blur(6px)}
.controls input[type=text]{flex:1;min-width:200px;background:#f5f6f8;color:var(--text);
border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:14px}
.controls select{background:#f5f6f8;color:var(--text);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;font-size:13px}
.controls label{color:var(--muted);font-size:13px;display:inline-flex;align-items:center;
gap:5px}
#cnt{color:var(--muted);font-size:13px;white-space:nowrap;margin-left:auto}
#cnt b{color:var(--text)}
.pager{display:flex;flex-wrap:wrap;gap:6px;align-items:center;justify-content:center;
margin:0 auto 16px;min-height:34px}
.pager button{background:#fff;color:var(--text);border:1px solid var(--line);
border-radius:8px;padding:6px 12px;font-size:13px;cursor:pointer}
.pager button:hover{border-color:var(--accent);color:var(--accent)}
.pager button.cur{background:var(--accent);color:#fff;border-color:var(--accent)}
.pager button:disabled{opacity:.4;cursor:default}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(210px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
overflow:hidden;box-shadow:var(--shadow);display:flex;flex-direction:column;
transition:transform .12s,border-color .12s,box-shadow .12s}
.card:hover{transform:translateY(-2px);border-color:var(--accent);
box-shadow:0 2px 6px rgba(24,34,60,.1),0 14px 34px rgba(24,34,60,.12)}
.card.hide{display:none}
.media{position:relative;flex:none;background:#f2f3f5;overflow:hidden}
.media a.media-link{display:block;width:100%;aspect-ratio:1/1}
.media img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .4s ease}
.card:hover .media img{transform:scale(1.04)}
.eye{position:absolute;top:8px;right:8px;width:32px;height:32px;border-radius:50%;
background:rgba(255,255,255,.92);border:1px solid var(--line);display:flex;
align-items:center;justify-content:center;opacity:0;transform:scale(.9);
transition:opacity .15s,transform .15s,background .15s}
.card:hover .eye{opacity:1;transform:none}
.eye:hover{background:var(--accent)}
.eye svg{width:16px;height:16px;stroke:#fff;fill:none;stroke-width:1.8;
stroke-linecap:round;stroke-linejoin:round}
.cat{position:absolute;left:0;bottom:8px;font-size:10px;font-weight:600;
text-transform:uppercase;letter-spacing:.4px;color:#fff;background:rgba(30,36,48,.78);
padding:3px 7px;border-radius:0 2px 2px 0}
.content{position:absolute;top:8px;left:8px;font-size:10px;font-weight:600;
padding:2px 7px;border-radius:9px;background:rgba(255,255,255,.92);
border:1px solid var(--line);white-space:nowrap;color:var(--text)}
.content.safe{color:#2e8b57}.content.nsfw{color:#d33}.content.unknown{color:var(--muted)}
.strip-row{display:flex;gap:6px;padding:8px 10px 0;align-items:center}
.strip{display:flex;gap:3px;align-items:center}
.strip-pal{flex:1;flex-wrap:wrap}
.strip-pose{flex:none}
.sw{width:20px;height:20px;border-radius:5px;border:1px solid var(--line)}
.strip-pose svg{display:block;border-radius:6px;border:1px solid var(--line)}
.cbody{padding:9px 11px 12px;display:flex;flex-direction:column;gap:7px}
.desc{font-size:13.5px;font-weight:600;line-height:1.32;color:var(--text);
word-break:break-word;display:-webkit-box;-webkit-line-clamp:2;
-webkit-box-orient:vertical;overflow:hidden}
.tagrow{display:flex;flex-wrap:wrap;gap:4px}
.tag{font-size:10.5px;color:var(--muted);background:#f0f2f6;
border:1px solid var(--line);border-radius:10px;padding:1px 7px}
.badge{display:inline-block;font-size:11px;font-weight:600;color:var(--good);
background:rgba(46,158,91,.12);border:1px solid rgba(46,158,91,.3);
border-radius:6px;padding:1px 6px;align-self:flex-start}
.badge.mid{color:var(--warn);background:rgba(201,150,42,.12);border-color:rgba(201,150,42,.3)}
.badge.low{color:var(--muted);background:rgba(139,147,165,.1);
border-color:rgba(139,147,165,.25)}
.actions{display:flex;gap:6px;margin-top:2px}
.actions a,.actions button{flex:1;text-align:center;border-radius:8px;padding:7px 8px;
font-size:11.5px;cursor:pointer;text-decoration:none;border:1px solid var(--line)}
.actions a{color:var(--text);background:#f0f2f6;font-weight:600}
.actions a:hover{border-color:var(--accent);color:var(--accent)}
.actions button{color:var(--muted);background:#f0f2f6;border:0}
.actions button:hover{color:var(--accent)}
.empty{color:var(--muted);text-align:center;padding:60px 0;font-size:15px;display:none}
.note{max-width:1440px;margin:26px auto 0;color:var(--muted);font-size:12.5px;
background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:12px 16px;box-shadow:var(--shadow)}
.note code{color:var(--text);background:#f0f2f6;padding:1px 5px;border-radius:5px}
"""


def _card(i: int, r: dict, thumbs: dict, has_sim: bool) -> str:
    path = r["path"]
    desc, category, tags = _meta(path)
    thumb = thumbs.get(str(r.get("hash", ""))) or html_mod.escape(str(r["thumb"]))
    score = r.get("score")
    cls = "low" if score is None else ("" if score >= 0.80 else ("mid" if score >= 0.55 else "low"))
    try:
        href = f"/open?hash={html_mod.escape(str(r.get('hash', '')))}"
    except Exception:
        href = "#"
    esc_desc = html_mod.escape(desc)
    esc_cat = html_mod.escape(category)
    esc_group = html_mod.escape(category.lower())
    score_txt = f"{score:.3f}" if score is not None else html_mod.escape(_fmt_size(r["size"]))
    badge_txt = f"{int(score * 100)}%" if score is not None else "—"
    sim = f'<span class="badge {cls}" data-orig="{html_mod.escape(score_txt)}">{badge_txt}</span>'
    sim_btn = (f'<button class="sim" data-hash="{html_mod.escape(str(r.get("hash", "")))}">Похожие</button>'
               ) if has_sim else ""
    tagrow = "".join(
        f'<span class="tag">{html_mod.escape(t)}</span>' for t in tags)
    content_badge = _content_badge(r)
    palette = _palette_squares(r)
    pose = _pose_squares(r)
    return f"""
<article class="card" data-hash="{html_mod.escape(str(r.get('hash', '')))}"
         data-name="{html_mod.escape(desc).lower()}" data-path="{html_mod.escape(path).lower()}"
         data-category="{esc_group}" data-size="{int(r['size'])}"
         data-mtime="{r.get('mtime', 0)}" data-i="{i}">
  <div class="media">
    <a class="media-link" href="{href}" target="_blank" title="Открыть оригинал">
      <img src="{thumb}" alt="{esc_desc}" loading="lazy">
    </a>
    <a class="eye" href="{href}" target="_blank" title="Открыть оригинал">
      <svg viewBox="0 0 24 24"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
    </a>
    <span class="cat" title="группа: {esc_cat}">{esc_cat}</span>
    {content_badge}
  </div>
  {palette if palette else ""}
  {pose if pose else ""}
  <div class="cbody">
    <div class="desc" title="{esc_desc}">{esc_desc}</div>
    {tagrow}
    {sim}
    <div class="actions">
      <a class="open" href="{href}" target="_blank">Открыть</a>
      {sim_btn}
    </div>
  </div>
</article>"""


def _group_options(rows: list) -> str:
    """Опции фильтра групп (категория из папки): value=lower, текст=оригинал."""
    cats = {}
    for r in rows:
        if not r.get("path"):
            continue
        cat = _meta(r["path"])[1] or ""
        if cat:
            cats[cat.lower()] = cat
    return "".join(
        f'<option value="{html_mod.escape(k)}">{html_mod.escape(v)}</option>'
        for k, v in sorted(cats.items()))


def render_browse(*, rows: list, image_vectors: dict, pose_vectors: dict,
                  face_vectors: dict, thumbs: dict, model_id: str,
                  dim: int, total_rows: int, db_dir: str, top_k: int,
                  has_sim: bool, has_pose: bool, has_face: bool) -> str:
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    cards = "\n".join(_card(i, r, thumbs, has_sim) for i, r in enumerate(rows))
    has_sim_js = "true" if has_sim else "false"
    has_pose_js = "true" if has_pose else "false"
    has_face_js = "true" if has_face else "false"
    pose_sel = " selected" if has_pose else " disabled"
    face_sel = " selected" if has_face else " disabled"
    image_sel = " selected" if not has_pose and not has_face else ""
    ivjson = json.dumps(image_vectors) if has_sim else "null"
    pvjson = json.dumps(pose_vectors) if has_pose else "null"
    fvjson = json.dumps(face_vectors) if has_face else "null"
    group_opts = _group_options(rows)
    js = _render_js(ivjson, pvjson, fvjson, top_k, has_sim_js, has_pose_js, has_face_js)
    sim_note = "" if has_sim else (
        "<span class=\"chip\">похожие: <b>отключены</b> "
        "(каталог &gt; " + str(config.BROWSE_MAX_EMBED) + " записей)</span>")
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Индекс изображений — {html_mod.escape(model_id)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Индекс изображений</h1>
  <div class="sub">
    <span>модель: <b>{html_mod.escape(model_id)}</b> ({dim}-dim)</span>
    <span>в инксе: <b>{total_rows}</b></span>
    <span>база: <b>{html_mod.escape(db_dir)}</b></span>
    <span>обновлено: <b>{now}</b></span>
  </div>
  <div class="chips">
    <span class="chip">метрика: <b>cosine</b></span>
    <span class="chip">топ похожих: <b>{top_k}</b></span>
    {sim_note}
    <span class="chip"><a href="#" onclick="openDup();return false;">Дубликаты →</a></span>
  </div>
</header>

<div class="controls">
  <input type="text" id="q" placeholder="Поиск по имени или пути…" autocomplete="off">
  <label for="group">Группа:</label>
  <select id="group"><option value="">все</option>{group_opts}</select>
  <label for="mode">Похожие:</label>
  <select id="mode">
    <option value="image"{image_sel}>изображения</option>
    <option value="pose"{pose_sel}>позы</option>
    <option value="face"{face_sel}>лица</option>
  </select>
  <label for="sort">Сорт:</label>
  <select id="sort">
    <option value="path">по имени</option>
    <option value="size_desc">по размеру ↓</option>
    <option value="size_asc">по размеру ↑</option>
    <option value="mtime_desc">по дате ↓</option>
  </select>
  <label for="perPage">на стр:</label>
  <select id="perPage">
    <option value="24">24</option>
    <option value="48">48</option>
    <option value="96">96</option>
    <option value="0">все</option>
  </select>
  <span id="cnt"></span>
</div>

<div class="pager" id="pagerTop"></div>

<main class="grid" id="grid">{cards}</main>
<div class="empty" id="empty">Ничего не найдено</div>

<div class="pager" id="pagerBottom"></div>

<div class="note">
  Клик по фото или глазик открывает оригинал через
  <code>/open?hash=</code> (сервер <code>imgsim serve</code>). В выпадающем
  «Похожие» выберите <code>изображения</code> / <code>позы</code> / <code>лица</code> —
  похожие ищутся по вектору модели прямо в браузере (cosine). Режим «позы» и
  «лица» доступен только на малых каталогах (вектора встроены в страницу).
  Под каждым фото — палитра и топ-5 точек позы.
</div>

<script>
{js}
</script>
</div>
</body>
</html>"""


def _render_js(ivjson: str, pvjson: str, fvjson: str, top_k: int,
               has_sim_js: str, has_pose_js: str, has_face_js: str) -> str:
    # Plain string (не f-string): JS-фигурные скобки не должны ломать
    # интерполяцию. Значения подставляются через уникальные токены.
    js = r"""
const VECTORS = __VECTORS__;
const POSE_V = __POSE_V__;
const FACE_V = __FACE_V__;
const TOP_K = __TOP_K__;
const HAS_SIM = __HAS_SIM__;
const HAS_POSE = __HAS_POSE__;
const HAS_FACE = __HAS_FACE__;
const PER_DEF = __PER_DEF__;
const cards=[...document.querySelectorAll('.card')];
const q=document.getElementById('q');
const group=document.getElementById('group');
const mode=document.getElementById('mode');
const sort=document.getElementById('sort');
const perPage=document.getElementById('perPage');
const cnt=document.getElementById('cnt');
const empty=document.getElementById('empty');
const pagerTop=document.getElementById('pagerTop');
const pagerBottom=document.getElementById('pagerBottom');
let MODE='image', PAGE=1, PER=PER_DEF;

function vecs(){
  if(MODE==='pose')return HAS_POSE?POSE_V:null;
  if(MODE==='face')return HAS_FACE?FACE_V:null;
  return HAS_SIM?VECTORS:null;
}
function matchCard(c){
  const term=q.value.trim().toLowerCase();
  const termHit=!term||c.dataset.name.includes(term)||c.dataset.path.includes(term);
  const groupHit=!group.value||c.dataset.group===group.value;
  return termHit&&groupHit;
}
function cmp(key){
  return (a,b)=>{
    switch(key){
      case 'size_desc':return +b.dataset.size-+a.dataset.size;
      case 'size_asc':return +a.dataset.size-+b.dataset.size;
      case 'mtime_desc':return +b.dataset.mtime-+a.dataset.mtime;
      default:return a.dataset.path.localeCompare(b.dataset.path);
    }
  };
}
function apply(){
  let list=cards.filter(matchCard);
  list.sort(cmp(sort.value));
  const total=list.length;
  PER=parseInt(perPage.value,10)||0;
  const per=PER>0?PER:list.length;
  const pages=Math.max(1,Math.ceil(total/per));
  if(PAGE>pages)PAGE=pages;
  const start=(PAGE-1)*per;
  const set=new Set(list.slice(start,start+per));
  for(const c of cards)c.classList.toggle('hide',!set.has(c));
  const shown=set.size;
  cnt.innerHTML='Показано <b>'+shown+'</b> из <b>'+total+'</b> · стр. '+PAGE+'/'+pages;
  empty.style.display=shown?'none':'block';
  renderPager(pages);
}
function renderPager(pages){
  let html='';
  html+='<button data-p="prev" '+(PAGE<=1?'disabled':'')+'>‹ Назад</button>';
  const win=4;
  let a=Math.max(1,PAGE-win), b=Math.min(pages,PAGE+win);
  if(a>1){html+='<button data-p="1">1</button>';if(a>2)html+='<span style="color:var(--muted)">…</span>';}
  for(let p=a;p<=b;p++){
    html+='<button data-p="'+p+'" class="'+(p===PAGE?'cur':'')+'">'+p+'</button>';
  }
  if(b<pages){if(b<pages-1)html+='<span style="color:var(--muted)">…</span>';
    html+='<button data-p="'+pages+'">'+pages+'</button>';}
  html+='<button data-p="next" '+(PAGE>=pages?'disabled':'')+'>Вперёд ›</button>';
  pagerTop.innerHTML=html;
  pagerBottom.innerHTML=html;
  pagerTop.querySelectorAll('button').forEach(b=>b.onclick=()=>{
    if(b.dataset.p==='prev')PAGE=Math.max(1,PAGE-1);
    else if(b.dataset.p==='next')PAGE=Math.min(pages,PAGE+1);
    else PAGE=parseInt(b.dataset.p,10);
    apply();
  });
}
function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
function clearBadges(){
  cards.forEach(c=>{
    const b=c.querySelector('.badge');
    if(b){b.className='badge low';b.textContent='—';}
  });
}
function findSimilar(m,hash){
  PAGE=1; PER=parseInt(perPage.value,10)||0;
  if(PER<=0){for(const c of cards)c.classList.remove('hide');}
  apply();
  const VS=vecs(); if(!VS)return;
  const vq=VS[hash]; if(!vq)return;
  const scored=Object.entries(VS)
    .filter(([h])=>h!==hash)
    .map(([h,v])=>({hash:h,score:dot(vq,v)}))
    .sort((a,b)=>b.score-a.score).slice(0,TOP_K);
  clearBadges();
  for(const s of scored){
    const c=[...cards.values()].find(x=>x.dataset.hash===s.hash);
    if(c){
      const badge=c.querySelector('.badge');
      if(badge){
        const cls=s.score>=.80?'':(s.score>=.55?'mid':'low');
        badge.className='badge'+cls;
        badge.textContent=Math.round(s.score*100)+'%';
      }
    }
  }
}
function openDup(){
  // file:// — дубликаты в той же папке что и страница; http (imgsim serve) — results/
  const p=(location.protocol==='file:')?'duplicates.html':'results/duplicates.html';
  window.open(p,'_blank');
}
q.addEventListener('input',()=>{PAGE=1;apply();});
group.addEventListener('change',()=>{PAGE=1;apply();});
sort.addEventListener('change',()=>{PAGE=1;apply();});
perPage.addEventListener('change',()=>{PAGE=1;apply();});
mode.addEventListener('change',()=>{MODE=mode.value;});
document.querySelectorAll('.open').forEach(a=>
  a.addEventListener('click',e=>{e.preventDefault();e.stopPropagation();
    window.open(a.href,'_blank');}));
document.querySelectorAll('.sim').forEach(b=>
  b.addEventListener('click',e=>{e.stopPropagation();findSimilar(MODE,b.dataset.hash);}));
apply();
"""
    return (js.replace("__VECTORS__", ivjson)
              .replace("__POSE_V__", pvjson)
              .replace("__FACE_V__", fvjson)
              .replace("__TOP_K__", str(top_k))
              .replace("__HAS_SIM__", has_sim_js)
              .replace("__HAS_POSE__", has_pose_js)
              .replace("__HAS_FACE__", has_face_js)
              .replace("__PER_DEF__", str(_PER_PAGE_DEFAULT)))


def run_browse(*, store, rows: list, embed: bool, top_k: int,
               model_id: str, dim: int, total_rows: int,
               out_path: str | None, log=print) -> Path:
    t0 = time.time()
    image_vectors = {r["hash"]: r["vector"] for r in rows if r.get("vector")} \
        if embed else {}
    pose_vectors = {}
    face_vectors = {}
    if embed:
        for r in rows:
            h = r.get("hash")
            if not h:
                continue
            p = r.get("pose")
            if p:
                try:
                    kps = json.loads(p).get("kps") or []
                    pose_vectors[h] = [v for pair in kps for v in pair]
                except Exception:
                    pass
            fv = r.get("face_vector")
            if fv:
                face_vectors[h] = fv
    has_sim = len(image_vectors) > 1
    has_pose = len(pose_vectors) > 1
    has_face = len(face_vectors) > 1
    out = Path(out_path) if out_path else \
        (config.results_dir() / f"browse_{datetime.now():%Y%m%d_%H%M%S}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    thumbs = {}
    for r in rows:
        try:
            thumbs[str(r["hash"])] = _data_uri(r["thumb"])
        except Exception:
            pass
    out.write_text(render_browse(
        rows=rows, image_vectors=image_vectors, pose_vectors=pose_vectors,
        face_vectors=face_vectors, thumbs=thumbs, model_id=model_id, dim=dim,
        total_rows=total_rows, db_dir=str(store.db_dir.resolve()),
        top_k=top_k, has_sim=has_sim, has_pose=has_pose, has_face=has_face),
        encoding="utf-8")
    dt = (time.time() - t0) * 1000
    log(f"Каталог: {total_rows} записей | поиск по имени/путю: да | "
        f"похожие: {'да' if has_sim else 'отключены'} | "
        f"позы: {'да' if has_pose else 'отключены'} | "
        f"лица: {'да' if has_face else 'отключены'} | группы: да ({dt:.0f} мс)")
    log(f"Галерея: {out.resolve()}")
    return out

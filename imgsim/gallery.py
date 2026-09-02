"""Самодостаточная HTML-галерея результатов поиска.

Миниатюры (WebP, ~320px) встраиваются base64 — файл открывается на любой
машине без доступа к исходной базе. Клик по карточке пытается открыть
оригинал (file://), рядом кнопка копирования пути.
"""

import base64
import html as html_mod
from datetime import datetime
from pathlib import Path

_CSS = """
:root{--bg:#0f1115;--panel:#171a21;--card:#1c2029;--line:#262b36;
--text:#e8eaf0;--muted:#8b93a5;--accent:#5b9dff;--good:#3fb970;--warn:#e3b341}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);
font:14px/1.45 -apple-system,'Segoe UI',Roboto,'Noto Sans',Arial,sans-serif;
padding:28px clamp(16px,4vw,48px) 60px}
header{max-width:1400px;margin:0 auto 22px}
h1{font-size:22px;font-weight:650;letter-spacing:.2px}
.sub{color:var(--muted);margin-top:6px;display:flex;flex-wrap:wrap;gap:8px 18px}
.sub b{color:var(--text);font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:20px;
padding:4px 12px;font-size:12.5px;color:var(--muted)}
.chip b{color:var(--accent)}
.querybox{display:flex;gap:14px;align-items:center;background:var(--panel);
border:1px solid var(--line);border-radius:14px;padding:12px 14px;
max-width:1400px;margin:0 auto 18px}
.querybox img{width:76px;height:76px;object-fit:cover;border-radius:10px;
border:1px solid var(--line)}
.qmeta{min-width:0}
.qname{font-weight:600;font-size:15px;word-break:break-all}
.qpath{color:var(--muted);font-size:12px;word-break:break-all;margin-top:3px}
.controls{max-width:1400px;margin:0 auto 18px;display:flex;flex-wrap:wrap;
gap:14px;align-items:center;background:var(--panel);
border:1px solid var(--line);border-radius:14px;padding:12px 16px;
position:sticky;top:10px;z-index:5;backdrop-filter:blur(6px)}
.controls label{color:var(--muted);font-size:13px}
.controls input[type=range]{width:220px;accent-color:var(--accent)}
#thrval{color:var(--accent);font-weight:600;min-width:44px;display:inline-block}
#cnt{color:var(--muted);font-size:13px;margin-left:auto}
#cnt b{color:var(--text)}
.grid{max-width:1400px;margin:0 auto;display:grid;gap:14px;
grid-template-columns:repeat(auto-fill,minmax(190px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
overflow:hidden;display:flex;flex-direction:column;transition:transform .12s,
border-color .12s}
.card:hover{transform:translateY(-2px);border-color:var(--accent)}
.card a.ph{display:block;position:relative}
.card img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block}
.badge{position:absolute;top:8px;right:8px;background:rgba(15,17,21,.82);
border:1px solid var(--line);border-radius:10px;padding:2px 8px;
font-size:12px;font-weight:650;color:var(--good)}
.badge.mid{color:var(--warn)}
.badge.low{color:var(--muted)}
.cbody{padding:9px 11px 11px;display:flex;flex-direction:column;gap:6px}
.fname{font-size:13px;font-weight:600;word-break:break-all;
display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
overflow:hidden}
.bar{height:4px;border-radius:2px;background:var(--line);overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent)}
.path{color:var(--muted);font-size:11px;word-break:break-all;
display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
overflow:hidden}
.copy{align-self:flex-start;background:var(--panel);color:var(--muted);
border:1px solid var(--line);border-radius:8px;padding:3px 9px;font-size:11px;
cursor:pointer}
.copy:hover{color:var(--text);border-color:var(--accent)}
.note{max-width:1400px;margin:26px auto 0;color:var(--muted);font-size:12.5px;
background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:12px 16px}
.note code{color:var(--text)}
.hidden{display:none}
"""

_JS = """
const slider=document.getElementById('thr');
const thrval=document.getElementById('thrval');
const cnt=document.getElementById('cnt');
const cards=[...document.querySelectorAll('.card')];
function apply(){
  const v=parseFloat(slider.value);
  thrval.textContent=v.toFixed(2).replace('0.','.').replace(/^([1-9])0+$/,'$1');
  let n=0;
  for(const c of cards){
    const s=parseFloat(c.dataset.score);
    const show=s>=v;
    c.classList.toggle('hidden',!show);
    if(show)n++;
  }
  cnt.innerHTML='Показано <b>'+n+'</b> из '+cards.length;
}
slider.addEventListener('input',apply);apply();
async function copyPath(btn,ev){
  ev.stopPropagation();
  const p=btn.dataset.path;
  let ok=true;
  try{await navigator.clipboard.writeText(p);}
  catch(e){
    try{const ta=document.createElement('textarea');ta.value=p;
      document.body.appendChild(ta);ta.select();
      document.execCommand('copy');ta.remove();}
    catch(e2){ok=false;}
  }
  const old=btn.textContent;
  btn.textContent=ok?'Путь скопирован':'Скопируйте вручную';
  setTimeout(()=>{btn.textContent=old;},1400);
}
document.querySelectorAll('.copy').forEach(
  b=>b.addEventListener('click',e=>copyPath(b,e)));
"""


def _b64_data_uri(path: str) -> str:
    try:
        data = Path(path).read_bytes()
        return "data:image/webp;base64," + base64.b64encode(data).decode()
    except Exception:
        # миниатюра могла быть удалена — подставляем прозрачный пиксель
        px = b"UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA=="
        return "data:image/webp;base64," + base64.b64encode(px).decode()


def _score_class(score: float) -> str:
    if score >= 0.80:
        return ""
    if score >= 0.55:
        return " mid"
    return " low"


def _card(i: int, r: dict) -> str:
    path = r["path"]
    name = Path(path).name
    score = r["score"]
    try:
        href = html_mod.escape(Path(path).as_uri())
    except Exception:
        href = "#"
    esc_name = html_mod.escape(name)
    esc_path = html_mod.escape(path)
    pct = round(score * 100)
    return f"""
<article class="card" data-score="{score:.4f}">
  <a class="ph" href="{href}" target="_blank" title="Открыть оригинал">
    <img src="{_b64_data_uri(r['thumb'])}" alt="{esc_name}" loading="lazy">
    <span class="badge{_score_class(score)}">{pct}%</span>
  </a>
  <div class="cbody">
    <div class="fname" title="{esc_name}">{esc_name}</div>
    <div class="bar"><i style="width:{pct}%"></i></div>
    <div class="path" title="{esc_path}">{esc_path}</div>
    <button class="copy" data-path="{esc_path}">Копировать путь</button>
  </div>
</article>"""


MODE_LABELS = {"image": "Похожие изображения", "face": "Похожие люди",
               "pose": "Похожие позы"}


def render_gallery(*, query_path: str, query_thumb: str, results: list,
                   model_id: str, variant: str, dim: int, total_rows: int,
                   emb_ms: float, srch_ms: float, min_score: float,
                   db_dir: str, mode: str = "image") -> str:
    qname = html_mod.escape(Path(query_path).name)
    qpath = html_mod.escape(query_path)
    title = MODE_LABELS.get(mode, MODE_LABELS["image"])
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    cards = "\n".join(_card(i, r) for i, r in enumerate(results))
    init_thr = round(min_score, 2)
    thr_attr = f'value="{init_thr}"'  # слайдер 0..1 со шагом 0.01

    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Похожие изображения — {qname}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">
    <span>запрос: <b>{qname}</b></span>
    <span>модель: <b>{html_mod.escape(model_id)}</b> ({dim}-dim)</span>
    <span>режим: <b>{html_mod.escape(MODE_LABELS.get(mode, mode))}</b></span>
    <span>в индксе: <b>{total_rows}</b></span>
    <span>эмбеддинг: <b>{emb_ms:.0f} мс</b></span>
    <span>поиск: <b>{srch_ms:.0f} мс</b></span>
    <span>{now}</span>
  </div>
  <div class="chips">
    <span class="chip">вариант: <b>{html_mod.escape(variant)}</b></span>
    <span class="chip">метрика: <b>cosine</b></span>
    <span class="chip">топ: <b>{len(results)}</b></span>
    <span class="chip">база: <b>{html_mod.escape(db_dir)}</b></span>
  </div>
</header>

<div class="querybox">
  <img src="{_b64_data_uri(query_thumb)}" alt="запрос">
  <div class="qmeta">
    <div class="qname">{qname}</div>
    <div class="qpath">{qpath}</div>
  </div>
</div>

<div class="controls">
  <label for="thr">Порог схожести:</label>
  <input type="range" id="thr" min="0" max="1" step="0.01" {thr_attr}>
  <span id="thrval">{init_thr:g}</span>
  <span id="cnt"></span>
</div>

<main class="grid" id="grid">
{cards}
</main>

<div class="note">
  Клик по изображению открывает оригинал через <code>file://</code> — некоторые
  браузеры (например, Chrome) могут блокировать такие переходы; в этом случае
  используйте кнопку «Копировать путь». Миниатюры встроены в файл — галерея
  не требует доступа к базе.
</div>

<script>{_JS}</script>
</body>
</html>"""

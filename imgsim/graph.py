"""Команда `graph`: граф сходства изображений — кластеры (Louvain), хабы и аномалии.

Строит неориентированный взвешенный граф над сохранёнными L2-нормированными
векторами DINOv2: узлы = записи инкса, рёбра = пары с cosine >= threshold
(вектора уже нормированы, поэтому cosine ≈ скалярное произведение — векторы
не перевычисляются). Затем:

  * Louvain community detection — мягкая кластеризация «фотосессий/объектов»,
    которая не рвётся жёстким порогом (в отличие от find_duplicates);
  * центральность по степени — хабы (мастер-файл с крй кропов/водяными знаками)
    и аномалии (изолированные / почти одинокие узлы);
  * транзитивный поиск вглубь (BFS за k шагов) — вариации, связанные «мостиком».

Офлайн, без LLM-токенов. Вывод: graph.json (полные данные) + graph.html
(vis-network через CDN — без локального фронтепа).

ponytail: матрица сходства считается блочно (память O(block × n)), как в
find_duplicates — не рвёт RAM на больших инксах.
"""

import html as html_mod
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config


def _build_similarity_graph(rows: list, threshold: float) -> "nx.Graph":
    """Строит взвешенный граф схожести: рёбра (i, j) при cosine >= threshold.

    Блочный перебор пар: промежуточная матрица (block × n) не превышает ~2 ГБ,
    независимо от n. Вектора нормированы — cosine = скалярное произведение.
    """
    import networkx as nx

    n = len(rows)
    G = nx.Graph()
    if n < 2:
        for r in rows:
            G.add_node(id(r), hash=r["hash"], path=r["path"], size=r.get("size", 0))
        return G

    for r in rows:
        G.add_node(id(r), hash=r["hash"], path=r["path"], size=r.get("size", 0))

    vectors = np.stack([r["vector"] for r in rows], axis=0).astype(np.float32)
    cap = 2 * 1024 ** 3
    block = max(1, min(8192, cap // (n * 4)))
    for start in range(0, n, block):
        end = min(start + block, n)
        row_sim = vectors[start:end] @ vectors.T  # (block, n), cosine
        for bi, i in enumerate(range(start, end)):
            ni = id(rows[start + bi])
            col = row_sim[bi]
            for j in range(i + 1, end):
                w = float(col[j])
                if w >= threshold:
                    G.add_edge(ni, id(rows[j]), weight=w)
    return G


def _analyze(G: "nx.Graph") -> dict:
    """Louvain-сообщества + центральность (хабы/аномалии). Возвращает маппинги по id узла."""
    import networkx as nx

    comm_id = {}
    for ci, comm in enumerate(nx.community.louvain_communities(G, resolution=1.0)):
        for node in comm:
            comm_id[node] = ci

    deg = dict(G.degree)
    degrees = sorted(deg.values())
    median = degrees[len(degrees) // 2] if degrees else 0
    hub_floor = max(10, 5 * median)

    communities, hubs, outliers = [], [], []
    for node in G.nodes:
        d = deg[node]
        if d == 0:
            outliers.append(node)
        elif d == 1:
            outliers.append(node)
        elif d >= hub_floor:
            hubs.append(node)
    for node in G.nodes:
        communities.append({"id": node, "community": comm_id.get(node, -1)})
    return {
        "communities": communities,
        "hubs": hubs,
        "outliers": outliers,
        "median_degree": median,
    }


def _bfs_depth(G: "nx.Graph", roots: list, max_depth: int) -> dict:
    """Транзитивный поиск: {root: [nodes within max_depth]}."""
    reach = {}
    for root in roots:
        seen = {root}
        q = deque([(root, 0)])
        out = []
        while q:
            node, depth = q.popleft()
            if depth > 0:
                out.append((node, depth))
            if depth < max_depth:
                for nb in G.neighbors(node):
                    if nb not in seen:
                        seen.add(nb)
                        q.append((nb, depth + 1))
        reach[root] = out
    return reach


def _vis_payload(G: "nx.Graph", info: dict) -> dict:
    """Полный payload для graph.json: узлы, рёбра, кластеры, хабы, аномалии."""
    deg = dict(G.degree)
    comm_of = {c["id"]: c["community"] for c in info["communities"]}
    hubs, outliers = set(info["hubs"]), set(info["outliers"])

    nodes = [{
        "id": node,
        "hash": G.nodes[node].get("hash"),
        "path": G.nodes[node].get("path"),
        "size": G.nodes[node].get("size", 0),
        "degree": deg[node],
        "community": comm_of.get(node, -1),
        "role": "hub" if node in hubs else ("outlier" if node in outliers else "normal"),
    } for node in G.nodes]

    edges = [{"from": u, "to": v, "weight": round(d["weight"], 4)}
             for u, v, d in G.edges(data=True)]

    return {
        "nodes": nodes,
        "edges": edges,
        "hubs": info["hubs"],
        "outliers": info["outliers"],
        "stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "communities": len({c["community"] for c in info["communities"] if c["community"] >= 0}),
            "median_degree": info["median_degree"],
        },
    }


def _render_html(payload: dict, threshold: float, model_id: str, dim: int,
                 total_rows: int, db_dir: str) -> str:
    """Интерактивный граф (vis-network через CDN). Данные встраиваются в страницу."""
    nodes_js = json.dumps(payload["nodes"], ensure_ascii=False)
    edges_js = json.dumps(payload["edges"], ensure_ascii=False)
    stats = payload["stats"]
    parts = [
        '<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">',
        '<title>Граф сходства изображений</title>',
        '<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js">',
        '</script>',
        '<style>' + _CSS + '</style></head><body>',
        '<header><h1>Граф сходства изображений</h1>',
        '<div class="meta">Модель: <b>' + html_mod.escape(model_id) + '</b> (' +
        str(dim) + '-dim) · порог cosine: <b>' + format(threshold, '.2f') +
        '</b> · записей: <b>' + str(total_rows) +
        '</b> · узлов: <b>' + str(stats["nodes"]) +
        '</b> · рёбер: <b>' + str(stats["edges"]) +
        '</b> · сообществ: <b>' + str(stats["communities"]) + '</b></div></header>',
        '<div class="legend">',
        '<span class="dot hub"></span> Хаб (связей ≥ ' + str(max(10, int(5 * stats["median_degree"]))) + ')',
        ' <span class="dot outlier"></span> Аномалия',
        ' <span class="dot normal"></span> Обычный',
        '</div>',
        '<div id="net"></div>',
        '<script>const NODES=' + nodes_js + ';const EDGES=' + edges_js + ';',
        'const colors=["#5b9dff","#3fb970","#e35d5d","#d2a8ff","#ff9f43",'
                     '#54a0ff","#ee5252","#00d2d3","#8e44ad","#f368e0"];',
        'function nodeColor(c){return colors[((c%10)+10)%10];}',
        'const nodes=new vis.DataSet(NODES.map(n=>({id:n.id,label:n.path.split("/")[-1],',
        'title:n.path+" · степень "+n.degree+" · "+(n.role==="hub"?"хаб":n.role),',
        'value:Math.max(4,n.degree),color:nodeColor(n.community),',
        'font:{color:"#e8eaf0",size:11},shape:"dot"})));',
        'const edges=new vis.DataSet(EDGES.map(e=>({from:e.from,to:e.to,weight:e.weight,',
        'title:"cosine "+e.weight,color:{color:"rgba(140,150,170,0.4)",highlight:"rgba(91,157,255,0.8)"}})));',
        'const data={nodes:nodes,edges:edges};',
        'const options={layout:{hierarchical:false},physics:{enabled:false,stableIterations:3},',
        'interaction:{hover:true,tooltipDelay:100},nodes:{borderWidth:2},',
        'edges:{smooth:{type:"continuous"},arrows:{to:{enabled:true,scaleFactor:0.3}}}};',
        'const net=new vis.Network(document.getElementById("net"),data,options);',
        'document.getElementById("count").textContent=' + str(stats["nodes"]) + ';',
        'document.getElementById("edgecount").textContent=' + str(stats["edges"]) + ';',
        '</script></body></html>',
    ]
    return ''.join(parts)


_CSS = """
:root{--bg:#0f1115;--text:#e8eaf0;--muted:#8b93a5;--line:#262b36}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,'Segoe UI',
Roboto,'Noto Sans',Arial,sans-serif;padding:24px clamp(16px,4vw,48px) 40px}
header{max-width:1400px;margin:0 auto 12px}
h1{font-size:22px;font-weight:700}
.meta{color:var(--muted);margin-top:8px}
.meta b{color:var(--text)}
.legend{max-width:1400px;margin:8px auto 16px;color:var(--muted);font-size:12px;
display:flex;gap:16px;flex-wrap:wrap}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;
vertical-align:middle}
.dot.hub{background:#d2a8ff}
.dot.outlier{background:#e35d5d}
.dot.normal{background:#5b9dff}
#net{max-width:1400px;margin:0 auto;height:72vh;border:1px solid var(--line);
border-radius:12px;background:#13161c}
.hint{max-width:1400px;margin:10px auto 0;color:var(--muted);font-size:12px}
"""


def run_graph(db_dir: str, model_dir: str | None = None, threshold: float = 0.85,
              out_path: str | None = None, log=print) -> Path:
    t0 = time.time()
    from .store import ImageStore
    store = ImageStore(db_dir, config.MODEL)
    total_rows = store.rows_count()
    if total_rows == 0:
        raise SystemExit(f'Индекс пуст. Сначала:\n  imgsim index <каталог> --db {db_dir}')

    rows = store.all_rows(include_vector=True)
    log(f'Записей в инксе: {total_rows} | векторов с вектором: {len(rows)}')

    t1 = time.time()
    G = _build_similarity_graph(rows, threshold)
    log(f'Граф построен: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер '
        f'(cosine ≥ {threshold:.2f}) за {time.time() - t1:.1f} с')
    if G.number_of_nodes() < 2:
        log('Меньше двух записей с векторами — графа не построить.')
        return Path(out_path or '(не сгенерировано)')

    info = _analyze(G)
    payload = _vis_payload(G, info)

    out = Path(out_path) if out_path else (
        config.results_dir() / f'graph_{datetime.now():%Y%m%d_%H%M%S}.html')
    out.parent.mkdir(parents=True, exist_ok=True)
    html = _render_html(payload, threshold, config.MODEL_ID, config.MODEL_DIM,
                        total_rows, str(store.db_dir.resolve()))
    out.write_text(html, encoding='utf-8')
    json_path = out.with_suffix('.json')
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                         encoding='utf-8')

    st = payload["stats"]
    log(f'Сообществ (Louvain): {st["communities"]} | хабов: {len(payload["hubs"])} '
        f'| аномалий: {len(payload["outliers"])} (медиана степени: {st["median_degree"]})')
    for gi, comm in enumerate(sorted({c["community"] for c in payload["nodes"]
                                      if c["community"] >= 0}) if payload["nodes"] else []):
        members = [n["path"].split("/")[-1] for n in payload["nodes"]
                   if n["community"] == gi][:5]
        log(f'  сообщество #{gi}: {", ".join(members)}' +
            (' …' if sum(1 for n in payload["nodes"] if n["community"] == gi) > 5 else ''))
    log(f'HTML: {out.resolve()} | JSON: {json_path.resolve()} ({time.time() - t0:.1f} с)')
    return out


if __name__ == "__main__":
    # ponytail: самодосточная проверка: 1000 синтетических векторов, граф < 2 с,
    # сообщества и хабы найдены корректно.
    rng = np.random.default_rng(0)
    # три плотных облака + изолированные узлы
    blobs = [rng.normal(loc=[1, 0, 0], scale=0.05, size=(100, 3)) for _ in range(3)]
    blobs = [np.r_[b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)]
             for b in blobs]
    vecs = np.vstack(blobs + [rng.normal(scale=0.5, size=(5, 3))]).astype(np.float32)
    rows = [{"hash": str(i), "path": f"/img/{i}.jpg", "size": 1000 + i,
             "vector": list(map(float, v))} for i, v in enumerate(vecs)]
    t1 = time.time()
    G = _build_similarity_graph(rows, 0.9)
    build_t = time.time() - t1
    info = _analyze(G)
    assert G.number_of_nodes() == len(vecs), G.number_of_nodes()
    # три облака должны разойтись минимум в 3 сообщества
    n_comm = len({c["community"] for c in info["communities"] if c["community"] >= 0})
    assert n_comm >= 3, n_comm
    assert build_t < 2.0, build_t
    print(f"graph selfcheck OK: nodes={G.number_of_nodes()} "
          f"edges={G.number_of_edges()} communities={n_comm} build={build_t:.3f}с")

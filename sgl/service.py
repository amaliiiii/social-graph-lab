"""Логика веб-сервиса без привязки к фреймворку (её проверяют тесты).

Состояния сервер не хранит: каждый запрос содержит источник графа, а
вычисленные результаты лежат в небольшом кэше в памяти процесса (без базы данных).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from collections import OrderedDict

import numpy as np

from . import higgs
from .cascade import Spread, cascade_size, mean_ci, simulate
from .graph import Graph
from .measures import METRIC_NAMES, compute_all

DATA_DIR = os.environ.get("SGL_DATA_DIR", "/data")
MAX_UPLOAD_EDGES = 200_000
CACHE_SIZE = 6

METRICS_INFO = [
    {"id": "in_degree", "label": "Входящая степень",
     "hint": "Сколько пользователей обращают на вас внимание: подписчики, репостеры, упоминающие."},
    {"id": "out_degree", "label": "Исходящая степень",
     "hint": "На скольких пользователей обращаете внимание вы сами."},
    {"id": "closeness", "label": "Близость",
     "hint": "Насколько близко, в среднем, находятся те, кто может донести до вас сообщение (по кратчайшим путям)."},
    {"id": "betweenness", "label": "Посредничество",
     "hint": "Через сколько кратчайших путей между другими проходит вершина: роль моста между группами."},
    {"id": "eigenvector", "label": "Собственный вектор",
     "hint": "Важна вершина, на которую указывают важные вершины."},
    {"id": "pagerank", "label": "PageRank",
     "hint": "Доля времени, которую случайный читатель проводит в вершине (d = 0,85)."},
    {"id": "hits_authority", "label": "Авторитетность (HITS)",
     "hint": "Велика, если на вершину указывают хорошие «наводчики»."},
    {"id": "coreness", "label": "Индекс k-ядра",
     "hint": "Номер слоя «луковицы»: чем больше, тем глубже вершина в плотной сердцевине."},
]
_METRIC_IDS = {m["id"] for m in METRICS_INFO}

DEMO = {
    "demo-g1": "Пример G1 из теории (8 вершин, неориентированный)",
    "demo-g2": "Пример G2 из теории (5 вершин, ориентированный)",
    "demo-social": "Учебная сеть (120 вершин, ориентированная)",
}
HIGGS_LABELS = {
    "social": "Higgs: подписки", "retweet": "Higgs: репосты",
    "reply": "Higgs: ответы", "mention": "Higgs: упоминания",
}

G1_EDGES = [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"), ("d", "e"),
            ("e", "f"), ("e", "g"), ("f", "g"), ("c", "h")]
G2_EDGES = [("B", "A"), ("C", "A"), ("D", "A"), ("D", "B"), ("E", "D"), ("A", "E"), ("C", "D")]


def _demo_social(n: int = 120, follows: int = 3, seed: int = 7) -> Graph:
    """Ориентированная сеть с предпочтительным присоединением: новые пользователи
    чаще подписываются на тех, у кого уже много подписчиков."""
    rng = np.random.default_rng(seed)
    indeg = np.ones(n)
    edges = []
    for u in range(1, n):
        k = min(follows, u)
        p = indeg[:u] / indeg[:u].sum()
        for v in rng.choice(u, size=k, replace=False, p=p):
            edges.append((u, int(v)))
            indeg[v] += 1
            if rng.random() < 0.15:          # иногда подписка взаимна
                edges.append((int(v), u))
                indeg[u] += 1
    return Graph.from_edges([(f"u{a}", f"u{b}") for a, b in edges], directed=True,
                            nodes=[f"u{i}" for i in range(n)])


def parse_edges_text(text: str, directed: bool = True) -> Graph:
    edges = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s,;]+", line)
        if len(parts) < 2:
            raise ValueError(f"не удалось разобрать строку: {line[:60]!r}")
        w = 1.0
        if len(parts) > 2:
            try:
                w = float(parts[2])
            except ValueError:
                w = 1.0
        edges.append((parts[0], parts[1], w))
        if len(edges) > MAX_UPLOAD_EDGES:
            raise ValueError(f"слишком много рёбер (допустимо до {MAX_UPLOAD_EDGES})")
    if not edges:
        raise ValueError("в тексте нет рёбер: нужны строки вида «u v» или «u v вес»")
    return Graph.from_edges(edges, directed=directed)


class Entry:
    def __init__(self, key: str, graph: Graph):
        self.key, self.graph = key, graph
        self._metrics = None
        self._modes = None
        self._spread = None
        self._index = None
        self._lock = threading.Lock()

    def metrics(self):
        with self._lock:
            if self._metrics is None:
                g = self.graph
                self._metrics, self._modes = compute_all(g, skip_paths=g.n > 5000)
            return self._metrics, self._modes

    def spread(self) -> Spread:
        with self._lock:
            if self._spread is None:
                self._spread = Spread(self.graph)
            return self._spread

    def index(self) -> dict[str, int]:
        with self._lock:
            if self._index is None:
                self._index = {self.graph.label(i): i for i in range(self.graph.n)}
            return self._index


_cache: "OrderedDict[str, Entry]" = OrderedDict()
_cache_lock = threading.RLock()


def list_sources() -> dict:
    items = [{"id": k, "label": v} for k, v in DEMO.items()]
    if os.path.isdir(DATA_DIR):
        for name in higgs.available_layers(DATA_DIR):
            items.append({"id": f"higgs-{name}", "label": HIGGS_LABELS[name]})
    return {"datasets": items}


def _load(source: dict) -> tuple[str, Graph]:
    ds, text = source.get("dataset"), source.get("edges_text")
    if ds:
        if ds == "demo-g1":
            return ds, Graph.from_edges(G1_EDGES, directed=False)
        if ds == "demo-g2":
            return ds, Graph.from_edges(G2_EDGES, directed=True)
        if ds == "demo-social":
            return ds, _demo_social()
        if ds.startswith("higgs-") and ds[6:] in higgs.LAYER_FILES:
            if ds[6:] not in higgs.available_layers(DATA_DIR):
                raise ValueError(f"файл слоя {ds[6:]} не найден в {DATA_DIR}")
            return ds, higgs.load_layer(DATA_DIR, ds[6:])
        raise ValueError(f"неизвестный набор данных: {ds}")
    if text:
        directed = bool(source.get("directed", True))
        key = "up:" + hashlib.sha1(f"{directed}|{text}".encode()).hexdigest()
        return key, parse_edges_text(text, directed)
    raise ValueError("не указан источник графа")


def get_entry(source: dict) -> Entry:
    ds, text = source.get("dataset"), source.get("edges_text")
    if ds:
        key = ds
    elif text:
        key = "up:" + hashlib.sha1(f"{bool(source.get('directed', True))}|{text}".encode()).hexdigest()
    else:
        raise ValueError("не указан источник графа")
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    key, graph = _load(source)          # тяжёлая загрузка вне блокировки
    with _cache_lock:
        entry = _cache.get(key) or Entry(key, graph)
        _cache[key] = entry
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
        return entry


def _num(x: float):
    return None if (x is None or isinstance(x, float) and math.isnan(x)) else round(float(x), 6)


def _view_indices(entry: Entry, max_nodes: int) -> np.ndarray:
    g = entry.graph
    if g.n <= max_nodes:
        return np.arange(g.n)
    pr = entry.metrics()[0]["pagerank"]
    return np.sort(np.argsort(-pr, kind="stable")[:max_nodes])


def analyze(source: dict, max_nodes: int = 150) -> dict:
    entry = get_entry(source)
    g = entry.graph
    metrics, modes = entry.metrics()
    idx = _view_indices(entry, max_nodes)
    pos = {int(v): i for i, v in enumerate(idx)}
    nodes = [{"label": g.label(int(v)),
              "m": {k: _num(metrics[k][v]) for k in METRIC_NAMES + ["hits_hub"]}}
             for v in idx]
    coo = g.A.tocoo()
    edges = []
    for u, v, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if u in pos and v in pos and (g.directed or u < v):
            edges.append([pos[u], pos[v], round(w, 4)])
    return {
        "key": entry.key, "n": g.n, "m": g.m, "directed": g.directed,
        "shown": len(nodes), "modes": modes,
        "metrics": METRICS_INFO, "nodes": nodes, "edges": edges,
    }


def _resolve(entry: Entry, labels) -> list[int]:
    idx = entry.index()
    out = []
    for s in labels:
        if str(s) not in idx:
            raise ValueError(f"нет вершины {s!r}")
        out.append(idx[str(s)])
    return out


def cascade(source: dict, seeds: list, model: str = "ic", p: float = 0.1,
            run: int = 0, max_nodes: int = 150) -> dict:
    if not seeds:
        raise ValueError("выберите хотя бы одну начальную вершину")
    if model not in ("ic", "wc", "lt"):
        raise ValueError("модель должна быть ic, wc или lt")
    if not 0.0 <= p <= 1.0:
        raise ValueError("вероятность должна быть в отрезке [0, 1]")
    entry = get_entry(source)
    g = entry.graph
    steps = simulate(entry.spread(), _resolve(entry, seeds), model, p, run)
    shown = set(int(v) for v in _view_indices(entry, max_nodes))
    out_steps = [{"nodes": [g.label(v) for v in st if v in shown], "new": len(st)}
                 for st in steps]
    return {"steps": out_steps, "total": cascade_size(steps), "n": g.n}


def compare(source: dict, metric: str, k: int, model: str = "ic", p: float = 0.1,
            runs: int = 100) -> dict:
    """Средний размер каскада для k лучших вершин по мере и для k случайных вершин.
    Оба набора запускаются в одних и тех же случайных «мирах»."""
    if metric not in _METRIC_IDS:
        raise ValueError(f"неизвестная мера: {metric}")
    entry = get_entry(source)
    g = entry.graph
    vals = entry.metrics()[0][metric]
    if np.isnan(vals).any():
        raise ValueError("эта мера для данного графа не вычислялась (граф слишком большой)")
    k = max(1, min(int(k), g.n, 50))
    runs = max(5, min(int(runs), 500 if g.n <= 5000 else 50))
    top = np.argsort(-vals, kind="stable")[:k].tolist()
    sp = entry.spread()
    rng = np.random.default_rng(12345)
    a, b = [], []
    for r in range(runs):
        rand = rng.choice(g.n, size=k, replace=False).tolist()
        a.append(cascade_size(simulate(sp, top, model, p, r)))
        b.append(cascade_size(simulate(sp, rand, model, p, r)))
    a, b = np.array(a, float), np.array(b, float)
    ma, ca = mean_ci(a)
    mb, cb = mean_ci(b)
    md, cd = mean_ci(a - b)
    return {"k": k, "runs": runs, "seeds": [g.label(v) for v in top],
            "top": {"mean": ma, "ci": ca}, "random": {"mean": mb, "ci": cb},
            "gain": {"mean": md, "ci": cd}, "ratio": ma / mb if mb > 0 else None}


def quiz(source: dict, node: str, metric: str, guess: float) -> dict:
    """Проверка ответа в режиме тренажёра."""
    if metric not in _METRIC_IDS:
        raise ValueError(f"неизвестная мера: {metric}")
    entry = get_entry(source)
    i = _resolve(entry, [node])[0]
    value = entry.metrics()[0][metric][i]
    if math.isnan(value):
        raise ValueError("мера не вычислялась для этого графа")
    tol = 0.01 if abs(value) <= 1 else 0.01 * abs(value)
    return {"correct": bool(abs(float(guess) - value) <= tol), "value": round(float(value), 4),
            "tolerance": float(tol)}

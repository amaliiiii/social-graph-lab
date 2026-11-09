"""Восемь мер центральности из части I доклада.

Все меры отвечают на вопрос «кто получает внимание / кто влиятелен» при
соглашении u -> v = «u обращает внимание на v».

  in_degree, out_degree   степени (раздел 2.2)
  closeness               близость по входящим путям (2.3)
  betweenness             посредничество, алгоритм Брандеса (2.4)
  eigenvector             центральность по собственному вектору (2.5)
  pagerank                PageRank, d = 0,85 (2.6)
  hits_authority          авторитетность по HITS (2.7); hits_hub — роль наводчика
  coreness                индекс k-ядра, алгоритм Батагеля и Заверншика (2.8)
"""
from __future__ import annotations

import numpy as np

from .graph import Graph

METRIC_NAMES = ["in_degree", "out_degree", "closeness", "betweenness",
                "eigenvector", "pagerank", "hits_authority", "coreness"]


# ---------------------------------------------------------------- степени
def in_degree(g: Graph) -> np.ndarray:
    return np.diff(g.AT.indptr).astype(float)


def out_degree(g: Graph) -> np.ndarray:
    return np.diff(g.A.indptr).astype(float)


# --------------------------------------------- близость и посредничество
def path_measures(g: Graph, pivots: int | None = None, seed: int = 0):
    """Близость и посредничество за один проход поиска в ширину из каждой
    начальной вершины (алгоритм Брандеса, O(nm)).

    pivots=None — точный расчёт из всех вершин. Иначе берётся выборка из
    `pivots` начальных вершин, результаты масштабируются на n/pivots (Brandes, Pich, 2007).

    Близость: C(v) = (r-1)/(n-1) * (r-1)/sum d(u, v), где суммирование по всем u,
    из которых v достижима (расстояния по входящим путям, как в NetworkX);
    при связном графе это (n-1)/sum d(u, v).
    Посредничество: сумма по упорядоченным парам; для неориентированного
    графа делится на 2 (т. е. считаются неупорядоченные пары).
    """
    n = g.n
    adj, radj = g.out_lists(), g.in_lists()
    if pivots is None or pivots >= n:
        sources, scale, exact = range(n), 1.0, True
    else:
        rng = np.random.default_rng(seed)
        sources = sorted(rng.choice(n, size=pivots, replace=False).tolist())
        scale, exact = n / pivots, False

    bc = [0.0] * n
    dsum = [0] * n
    reach = [0] * n
    for s in sources:
        dist = [-1] * n
        sigma = [0] * n
        dist[s], sigma[s] = 0, 1
        order = [s]
        head = 0
        while head < len(order):
            v = order[head]
            head += 1
            dv = dist[v] + 1
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dv
                    order.append(w)
                if dist[w] == dv:
                    sigma[w] += sigma[v]
        delta = [0.0] * n
        for idx in range(len(order) - 1, 0, -1):
            w = order[idx]
            coef = (1.0 + delta[w]) / sigma[w]
            dw = dist[w] - 1
            for v in radj[w]:
                if dist[v] == dw:
                    delta[v] += sigma[v] * coef
            bc[w] += delta[w]
            dsum[w] += dist[w]
            reach[w] += 1

    bc = np.array(bc) * scale
    if not g.directed:
        bc /= 2.0
    reach = np.array(reach, dtype=float)
    dsum = np.array(dsum, dtype=float)
    if not exact:
        reach = np.minimum(reach * scale, n - 1)
        dsum = dsum * scale
    if n > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            close = np.where(dsum > 0, (reach / (n - 1)) * (reach / dsum), 0.0)
    else:
        close = np.zeros(n)
    return close, bc


# ------------------------------------------------- собственный вектор
def eigenvector(g: Graph, tol: float = 1e-12, max_iter: int = 20000) -> np.ndarray:
    """Степенной метод для x = A^T x (важна вершина, на которую указывают важные).
    Добавление единичной матрицы (A^T + I) не меняет собственных векторов, но
    убирает колебания на двудольных графах. Нормировка: максимум равен 1."""
    n = g.n
    if n == 0:
        return np.zeros(0)
    AT = g.AT
    x = np.ones(n)
    for _ in range(max_iter):
        y = AT @ x + x
        top = y.max()
        if top <= 0:
            return np.zeros(n)
        y /= top
        if np.abs(y - x).sum() < tol:
            x = y
            break
        x = y
    return x


# ------------------------------------------------------------ PageRank
def pagerank(g: Graph, damping: float = 0.85, tol: float = 1e-12,
             max_iter: int = 2000) -> np.ndarray:
    """PR(v) = (1-d)/n + d * sum PR(u) w(u,v) / W(u); вершины без исходящих
    рёбер равномерно раздают свой вес всем вершинам."""
    n = g.n
    if n == 0:
        return np.zeros(0)
    outw = np.asarray(g.A.sum(axis=1)).ravel()
    dang = outw == 0
    inv = np.zeros(n)
    inv[~dang] = 1.0 / outw[~dang]
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        xn = damping * (g.AT @ (x * inv)) + (damping * x[dang].sum() + 1.0 - damping) / n
        done = np.abs(xn - x).sum() < tol
        x = xn
        if done:
            break
    return x / x.sum()


# ---------------------------------------------------------------- HITS
def hits(g: Graph, tol: float = 1e-12, max_iter: int = 20000):
    """Авторитетность a и роль наводчика h (Клейнберг, 1999), L2-нормировка."""
    n = g.n
    a = np.zeros(n)
    h = np.ones(n)
    for _ in range(max_iter):
        a_new = g.AT @ h
        na = np.linalg.norm(a_new)
        if na == 0:
            return np.zeros(n), np.zeros(n)
        a_new /= na
        h_new = g.A @ a_new
        nh = np.linalg.norm(h_new)
        if nh == 0:
            return a_new, np.zeros(n)
        h_new /= nh
        done = np.abs(a_new - a).sum() + np.abs(h_new - h).sum() < tol
        a, h = a_new, h_new
        if done:
            break
    return a, h


# ------------------------------------------------------------ k-ядра
def coreness(g: Graph) -> np.ndarray:
    """Алгоритм Батагеля и Заверншика, O(n + m); направление рёбер игнорируется."""
    n = g.n
    if n == 0:
        return np.zeros(0)
    S = g.symmetric_pattern()
    ip, ix = S.indptr.tolist(), S.indices.tolist()
    deg = [ip[i + 1] - ip[i] for i in range(n)]
    md = max(deg)
    bins = [0] * (md + 1)
    for d in deg:
        bins[d] += 1
    start = 0
    for d in range(md + 1):
        num = bins[d]
        bins[d] = start
        start += num
    pos = [0] * n
    vert = [0] * n
    for v in range(n):
        pos[v] = bins[deg[v]]
        vert[pos[v]] = v
        bins[deg[v]] += 1
    for d in range(md, 0, -1):
        bins[d] = bins[d - 1]
    bins[0] = 0
    for i in range(n):
        v = vert[i]
        for j in range(ip[v], ip[v + 1]):
            u = ix[j]
            if deg[u] > deg[v]:
                du, pu = deg[u], pos[u]
                pw = bins[du]
                w = vert[pw]
                if u != w:
                    pos[u], pos[w] = pw, pu
                    vert[pu], vert[pw] = w, u
                bins[du] += 1
                deg[u] -= 1
    return np.array(deg, dtype=float)


# ------------------------------------------------------------- всё вместе
def compute_all(g: Graph, exact_limit: int = 400, pivots: int = 150,
                seed: int = 0, skip_paths: bool = False):
    """Возвращает (метрики, режимы): словарь имя -> массив и словарь имя -> режим
    расчёта для близости и посредничества ('exact' | 'sampled' | 'skipped')."""
    out = {
        "in_degree": in_degree(g),
        "out_degree": out_degree(g),
        "eigenvector": eigenvector(g),
        "pagerank": pagerank(g),
        "coreness": coreness(g),
    }
    a, h = hits(g)
    out["hits_authority"], out["hits_hub"] = a, h
    if skip_paths:
        out["closeness"] = np.full(g.n, np.nan)
        out["betweenness"] = np.full(g.n, np.nan)
        mode = "skipped"
    else:
        exact = g.n <= exact_limit
        out["closeness"], out["betweenness"] = path_measures(
            g, None if exact else pivots, seed)
        mode = "exact" if exact else "sampled"
    modes = {"closeness": mode, "betweenness": mode}
    return out, modes

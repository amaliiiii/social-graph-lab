"""Эксперименты 1–4 из части III.

Учебная синтетическая сеть:   python scripts/run_experiments.py --n 1500 --out results.json
Слои Higgs (эксперименты 1–3): python scripts/run_experiments.py --data ./data --layer social --pivots 100 --skip-paths

Эксперимент 2 (связь с откликом) выполняется, если в --data есть слои social и retweet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgl import higgs                                     # noqa: E402
from sgl.cascade import Spread, cascade_size, mean_ci, simulate   # noqa: E402
from sgl.graph import Graph                               # noqa: E402
from sgl.measures import (METRIC_NAMES, compute_all, coreness, eigenvector, hits,   # noqa: E402
                          in_degree, pagerank, path_measures)


def synthetic(n: int, follows: int = 3, seed: int = 7) -> Graph:
    """Ориентированная сеть с предпочтительным присоединением (быстрая версия)."""
    rng = np.random.default_rng(seed)
    ends = [0]                     # список концов рёбер: выбор элемента пропорционален входящей степени + 1
    src, dst = [], []
    for u in range(1, n):
        chosen = set()
        while len(chosen) < min(follows, u):
            chosen.add(ends[int(rng.integers(len(ends)))] if rng.random() < 0.85 else int(rng.integers(u)))
        for v in chosen:
            src.append(u); dst.append(v); ends.append(v)
            if rng.random() < 0.25:        # взаимная подписка: в графе появляются циклы
                src.append(v); dst.append(u); ends.append(u)
        ends.append(u)
    return Graph(n, src, dst)


def jaccard_top(a, b, N):
    ta, tb = set(np.argsort(-a, kind="stable")[:N]), set(np.argsort(-b, kind="stable")[:N])
    return len(ta & tb) / len(ta | tb)


def exp1(metrics, names):
    rho, tau, jac = {}, {}, {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = metrics[a], metrics[b]
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            rho[f"{a}|{b}"] = float(spearmanr(x, y)[0])
            tau[f"{a}|{b}"] = float(kendalltau(x, y)[0])
            jac[f"{a}|{b}"] = {str(N): jaccard_top(x, y, N) for N in (10, 100)}
    return {"spearman": rho, "kendall_tau_b": tau, "top_overlap": jac}


def exp2(layers, metrics_social):
    r = in_degree(layers["retweet"])
    mask = (r > 0) | (in_degree(layers["social"]) > 0)
    out = {}
    for name in METRIC_NAMES:
        x = metrics_social[name]
        if np.isnan(x).any():
            continue
        out[name] = {"spearman": float(spearmanr(x[mask], r[mask])[0]),
                     "kendall_tau_b": float(kendalltau(x[mask], r[mask])[0])}
    return out


def exp3(g, metrics, k, p, runs, model="ic"):
    sp = Spread(g)
    rng = np.random.default_rng(12345)
    rand_sets = [rng.choice(g.n, size=k, replace=False).tolist() for _ in range(runs)]
    base = np.array([cascade_size(simulate(sp, rand_sets[r], model, p, r)) for r in range(runs)], float)
    mb, cb = mean_ci(base)
    res = {"random": {"mean": mb, "ci": cb}}
    for name in METRIC_NAMES:
        vals = metrics[name]
        if np.isnan(vals).any():
            continue
        top = np.argsort(-vals, kind="stable")[:k].tolist()
        sizes = np.array([cascade_size(simulate(sp, top, model, p, r)) for r in range(runs)], float)
        m, c = mean_ci(sizes)
        md, cd = mean_ci(sizes - base)
        res[name] = {"mean": m, "ci": c, "gain": md, "gain_ci": cd, "ratio": m / mb if mb else None}
    return res


def timeit(fn, repeats=3):
    best = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); best.append(time.perf_counter() - t)
    return float(np.median(best))


def exp4(sizes_cheap, sizes_path, follows):
    cheap = {"coreness": coreness, "pagerank": pagerank, "hits": hits, "eigenvector": eigenvector}
    res = {name: [] for name in cheap}
    res["m_edges"] = []
    for n in sizes_cheap:
        g = synthetic(n, follows)
        res["m_edges"].append(int(g.A.nnz))
        for name, fn in cheap.items():
            res[name].append(timeit(lambda: fn(g)))
    path = {"betweenness_exact": [], "betweenness_100_pivots": []}
    for n in sizes_path:
        g = synthetic(n, 5)
        path["betweenness_exact"].append(timeit(lambda: path_measures(g), 1))
        path["betweenness_100_pivots"].append(timeit(lambda: path_measures(g, pivots=min(100, n)), 1))
    slopes = {}
    for name in cheap:
        slopes[name] = float(np.polyfit(np.log(sizes_cheap), np.log(res[name]), 1)[0])
    for name, ts in path.items():
        slopes[name] = float(np.polyfit(np.log(sizes_path), np.log(ts), 1)[0])
    return {"sizes_cheap": sizes_cheap, "sizes_path": sizes_path, "cheap": res, "path": path, "slopes": slopes}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="каталог Higgs; без него берётся синтетическая сеть")
    ap.add_argument("--layer", default="social")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--p", type=float, nargs="+", default=[0.05, 0.1])
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--pivots", type=int, default=100)
    ap.add_argument("--skip-paths", action="store_true")
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    layers = None
    if a.data:
        layers = higgs.load_layers(a.data)
        g = layers[a.layer]
    else:
        g = synthetic(a.n)
    print(f"граф: n={g.n}, m={g.m}")
    t = time.time()
    metrics, modes = compute_all(g, exact_limit=a.n if not a.data else 400, pivots=a.pivots,
                                 skip_paths=a.skip_paths)
    print(f"меры: {time.time() - t:.1f} с, режим путей: {modes['betweenness']}")
    out = {"n": g.n, "m": g.m, "modes": modes}
    names = METRIC_NAMES
    out["exp1"] = exp1(metrics, names)
    if layers and "social" in layers and "retweet" in layers:
        out["exp2"] = exp2(layers, compute_all(layers["social"], skip_paths=True)[0]
                           if a.layer != "social" else metrics)
    out["exp3"] = {str(p): exp3(g, metrics, a.k, p, a.runs) for p in a.p}
    if not a.skip_timing:
        out["exp4"] = exp4([1000, 10000, 100000], [100, 200, 400, 800], 10)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("записано:", a.out)


if __name__ == "__main__":
    main()

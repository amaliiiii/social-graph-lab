"""Модели каскадов: независимые каскады (IC), взвешенный каскад (WC), линейный порог (LT).

Распространение идёт по транспонированному графу: если u подписан на v
(ребро u -> v), то сообщение v может дойти до u.

Случайность задаётся «мирами»: номер запуска `run` вместе с номером ребра (или вершины)
однозначно определяет случайное число (хеш splitmix64). Поэтому разные начальные
множества, запущенные с одним и тем же `run`, сравниваются на одних и тех же живых
рёбрах (общие случайные числа, часть III, эксперимент 3).
"""
from __future__ import annotations

from itertools import product

import numpy as np

from .graph import Graph

_M = (1 << 64) - 1


def _mix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _M
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M
    return x ^ (x >> 31)


def uniform(run: int, key: int) -> float:
    """Детерминированное число из [0, 1) по паре (запуск, ребро/вершина)."""
    return _mix(_mix(run) ^ (key + 0x632BE59BD9B4E019)) / 2.0 ** 64


class Spread:
    """Граф распространения: из v информация идёт к подписчикам v."""

    def __init__(self, g: Graph):
        self.n = g.n
        AT = g.AT                      # строка v: те u, у кого есть ребро u -> v
        self.indptr = AT.indptr.tolist()
        self.indices = AT.indices.tolist()
        outdeg = np.diff(g.A.indptr)   # на скольких пользователей подписан u
        self.inv_out = [1.0 / d if d > 0 else 0.0 for d in outdeg]
        self.m = len(self.indices)

    def edge_prob(self, u: int, model: str, p: float) -> float:
        return self.inv_out[u] if model == "wc" else p


def simulate(sp: Spread, seeds, model: str = "ic", p: float = 0.1, run: int = 0):
    """Одна симуляция. Возвращает список шагов: steps[0] — начальные вершины,
    steps[t] — вершины, активированные на шаге t."""
    if model not in ("ic", "wc", "lt"):
        raise ValueError("модель должна быть 'ic', 'wc' или 'lt'")
    active = [False] * sp.n
    first = []
    for s in seeds:
        if not active[s]:
            active[s] = True
            first.append(s)
    steps = [first]
    frontier = first
    ip, ix = sp.indptr, sp.indices
    if model in ("ic", "wc"):
        while frontier:
            nxt = []
            for v in frontier:
                for j in range(ip[v], ip[v + 1]):
                    u = ix[j]
                    if not active[u] and uniform(run, j) < sp.edge_prob(u, model, p):
                        active[u] = True
                        nxt.append(u)
            if nxt:
                steps.append(nxt)
            frontier = nxt
    else:
        theta = {}
        acc = {}
        while frontier:
            touched = []
            for v in frontier:
                for j in range(ip[v], ip[v + 1]):
                    u = ix[j]
                    if not active[u]:
                        if u not in acc:
                            acc[u] = 0.0
                            theta[u] = uniform(run, (1 << 40) + u)
                        acc[u] += sp.inv_out[u]
                        touched.append(u)
            nxt = []
            for u in dict.fromkeys(touched):
                if not active[u] and acc[u] >= theta[u] - 1e-12:
                    active[u] = True
                    nxt.append(u)
            if nxt:
                steps.append(nxt)
            frontier = nxt
    return steps


def cascade_size(steps) -> int:
    return sum(len(s) for s in steps)


def monte_carlo(sp: Spread, seeds, model: str = "ic", p: float = 0.1,
                runs: int = 200, base: int = 0) -> np.ndarray:
    """Размеры каскадов в `runs` мирах base, base+1, ... ."""
    return np.array([cascade_size(simulate(sp, seeds, model, p, base + r))
                     for r in range(runs)], dtype=float)


def mean_ci(x: np.ndarray):
    """Среднее и половина ширины 95-процентного интервала (нормальное приближение)."""
    if len(x) < 2:
        return float(np.mean(x)), 0.0
    return float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x)))


def exact_ic_spread(sp: Spread, seeds, model: str = "ic", p: float = 0.1) -> float:
    """Точное σ(S) перебором всех 2^m наборов живых рёбер (только для m <= 20).
    Нужно для проверки симулятора и свойства субмодулярности."""
    edges = [(v, sp.indices[j], sp.edge_prob(sp.indices[j], model, p))
             for v in range(sp.n) for j in range(sp.indptr[v], sp.indptr[v + 1])]
    if len(edges) > 20:
        raise ValueError("слишком много рёбер для точного перебора")
    total = 0.0
    for mask in product((0, 1), repeat=len(edges)):
        prob = 1.0
        adj: dict[int, list[int]] = {}
        for live, (v, u, pe) in zip(mask, edges):
            prob *= pe if live else 1.0 - pe
            if live:
                adj.setdefault(v, []).append(u)
        if prob == 0.0:
            continue
        seen = set(seeds)
        stack = list(seen)
        while stack:
            v = stack.pop()
            for u in adj.get(v, ()):
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
        total += prob * len(seen)
    return total

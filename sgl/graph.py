"""Ориентированный взвешенный граф на разреженных матрицах SciPy."""
from __future__ import annotations

from typing import Hashable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp


class Graph:
    """Граф с n вершинами 0..n-1.

    A[u, v] = вес ребра u -> v («u обращает внимание на v»).
    Петли удаляются, параллельные рёбра в ориентированном графе складываются
    (вес = число повторений). Неориентированный граф хранится как симметричный.
    """

    def __init__(self, n: int, src, dst, w=None, labels: Sequence | None = None,
                 directed: bool = True):
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        w = np.ones(len(src)) if w is None else np.asarray(w, dtype=float)
        if not (len(src) == len(dst) == len(w)):
            raise ValueError("src, dst и w должны быть одной длины")
        if len(src) and (src.min() < 0 or dst.min() < 0 or max(src.max(), dst.max()) >= n):
            raise ValueError("номер вершины выходит за пределы 0..n-1")
        keep = src != dst                      # петли удаляем
        src, dst, w = src[keep], dst[keep], w[keep]
        if not directed and len(src):
            lo, hi = np.minimum(src, dst), np.maximum(src, dst)
            _, first = np.unique(lo * n + hi, return_index=True)   # одно ребро на пару
            lo, hi, w = lo[first], hi[first], w[first]
            src, dst, w = np.concatenate([lo, hi]), np.concatenate([hi, lo]), np.concatenate([w, w])
        self.n = int(n)
        self.directed = bool(directed)
        self.A = sp.coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()   # дубликаты суммируются
        self.A.sort_indices()
        self.labels = list(labels) if labels is not None else None
        self._AT = None
        self._out = None
        self._in = None

    # ---------- построение ----------
    @classmethod
    def from_edges(cls, edges: Iterable[Sequence], directed: bool = True,
                   nodes: Iterable[Hashable] | None = None) -> "Graph":
        """Граф из списка рёбер (u, v) или (u, v, вес); метки вершин — любые хешируемые."""
        index: dict[Hashable, int] = {}
        labels: list[str] = []

        def idx(x: Hashable) -> int:
            if x not in index:
                index[x] = len(labels)
                labels.append(str(x))
            return index[x]

        if nodes is not None:
            for x in nodes:
                idx(x)
        src, dst, w = [], [], []
        for e in edges:
            src.append(idx(e[0]))
            dst.append(idx(e[1]))
            w.append(float(e[2]) if len(e) > 2 else 1.0)
        return cls(len(labels), src, dst, w, labels, directed)

    # ---------- свойства ----------
    @property
    def AT(self) -> sp.csr_matrix:
        if self._AT is None:
            self._AT = self.A.T.tocsr()
            self._AT.sort_indices()
        return self._AT

    @property
    def m(self) -> int:
        """Число рёбер (для неориентированного графа — число неориентированных рёбер)."""
        return self.A.nnz if self.directed else self.A.nnz // 2

    def label(self, i: int) -> str:
        return str(self.labels[i]) if self.labels is not None else str(i)

    def out_lists(self) -> list[list[int]]:
        """Кого вершина «видит»: списки исходящих соседей (кэшируются)."""
        if self._out is None:
            ip, ix = self.A.indptr, self.A.indices
            self._out = [ix[ip[i]:ip[i + 1]].tolist() for i in range(self.n)]
        return self._out

    def in_lists(self) -> list[list[int]]:
        if self._in is None:
            ip, ix = self.AT.indptr, self.AT.indices
            self._in = [ix[ip[i]:ip[i + 1]].tolist() for i in range(self.n)]
        return self._in

    def symmetric_pattern(self) -> sp.csr_matrix:
        """Простой неориентированный граф без весов (для k-ядер)."""
        S = (self.A + self.AT).tocsr()
        S.data[:] = 1.0
        S.sort_indices()
        return S

    def transpose(self) -> "Graph":
        coo = self.AT.tocoo()
        return Graph(self.n, coo.row, coo.col, coo.data, self.labels, self.directed)

    def union(self, other: "Graph", weight: float = 1.0) -> "Graph":
        """Взвешенная сумма двух графов на одном множестве вершин."""
        if self.n != other.n:
            raise ValueError("графы должны быть заданы на одном множестве вершин")
        M = (self.A + weight * other.A).tocoo()
        return Graph(self.n, M.row, M.col, M.data, self.labels, self.directed and other.directed)

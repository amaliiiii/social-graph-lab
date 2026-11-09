"""Проверка мер на примерах G1 и G2 из части I (значения посчитаны вручную)."""
import unittest

import numpy as np

from sgl.graph import Graph
from sgl.measures import compute_all, coreness, hits, pagerank, path_measures
from sgl.service import G1_EDGES, G2_EDGES


def by_label(g, arr):
    return {g.label(i): float(arr[i]) for i in range(g.n)}


class TestG1(unittest.TestCase):
    """Неориентированный граф: две «компании» a-b-c и e-f-g, цепочка c-d-e, лист h при c."""

    @classmethod
    def setUpClass(cls):
        cls.g = Graph.from_edges(G1_EDGES, directed=False)
        cls.m, cls.modes = compute_all(cls.g)

    def test_degree(self):
        d = by_label(self.g, self.m["in_degree"])
        self.assertEqual(d, {"a": 2, "b": 2, "c": 4, "d": 2, "e": 3, "f": 2, "g": 2, "h": 1})

    def test_closeness(self):
        c = by_label(self.g, self.m["closeness"])
        self.assertAlmostEqual(c["c"], 7 / 12, places=9)
        self.assertAlmostEqual(c["d"], 7 / 12, places=9)
        self.assertAlmostEqual(c["e"], 0.5, places=9)
        self.assertAlmostEqual(c["a"], 7 / 17, places=9)
        self.assertAlmostEqual(c["h"], 7 / 18, places=9)
        self.assertAlmostEqual(c["f"], 7 / 19, places=9)

    def test_betweenness(self):
        b = by_label(self.g, self.m["betweenness"])
        self.assertEqual((b["c"], b["d"], b["e"]), (14.0, 12.0, 10.0))
        for v in "abfgh":
            self.assertEqual(b[v], 0.0)

    def test_eigenvector(self):
        e = by_label(self.g, self.m["eigenvector"])
        expected = {"c": 1.0, "a": 0.69, "b": 0.69, "d": 0.66, "e": 0.62, "f": 0.43, "g": 0.43, "h": 0.41}
        for k, v in expected.items():
            self.assertAlmostEqual(e[k], v, delta=0.006, msg=k)

    def test_coreness(self):
        k = by_label(self.g, self.m["coreness"])
        self.assertEqual(k["h"], 1)
        for v in "abcdefg":
            self.assertEqual(k[v], 2)

    def test_pagerank_sums_to_one(self):
        self.assertAlmostEqual(float(self.m["pagerank"].sum()), 1.0, places=9)


class TestG2(unittest.TestCase):
    """Ориентированный граф: ребро u -> v = «u обращает внимание на v»."""

    @classmethod
    def setUpClass(cls):
        cls.g = Graph.from_edges(G2_EDGES, directed=True)
        cls.m, _ = compute_all(cls.g)

    def test_degrees(self):
        self.assertEqual(by_label(self.g, self.m["in_degree"]), {"B": 1, "A": 3, "C": 0, "D": 2, "E": 1})
        self.assertEqual(by_label(self.g, self.m["out_degree"]), {"B": 1, "A": 1, "C": 2, "D": 2, "E": 1})

    def test_pagerank(self):
        pr = by_label(self.g, self.m["pagerank"])
        for k, v in {"A": 0.282, "D": 0.272, "E": 0.270, "B": 0.146}.items():
            self.assertAlmostEqual(pr[k], v, delta=0.0006, msg=k)
        self.assertAlmostEqual(pr["C"], 0.15 / 5, places=9)      # только случайные перескоки

    def test_hits(self):
        a, h = hits(self.g)
        a, h = by_label(self.g, a), by_label(self.g, h)
        for k, v in {"A": 0.844, "D": 0.449, "B": 0.293, "C": 0.0, "E": 0.0}.items():
            self.assertAlmostEqual(a[k], v, delta=0.001, msg="авторитет " + k)
        for k, v in {"C": 0.657, "D": 0.577, "B": 0.429, "E": 0.228, "A": 0.0}.items():
            self.assertAlmostEqual(h[k], v, delta=0.001, msg="наводчик " + k)


class TestAgainstNetworkX(unittest.TestCase):
    """Сверка с библиотекой NetworkX на случайных графах (пропускается без неё)."""

    def setUp(self):
        try:
            import networkx as nx
        except ImportError:
            self.skipTest("NetworkX не установлен")
        self.nx = nx

    def _random_graph(self, directed, seed):
        rng = np.random.default_rng(seed)
        n = 40
        edges = {(int(a), int(b)) for a, b in rng.integers(0, n, size=(160, 2)) if a != b}
        return Graph.from_edges(sorted(edges), directed=directed, nodes=range(n)), edges, n

    def test_directed(self):
        nx = self.nx
        g, edges, n = self._random_graph(True, 1)
        G = nx.DiGraph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
        m, _ = compute_all(g)
        order = [int(l) for l in g.labels]
        pr = nx.pagerank(G, alpha=0.85, tol=1e-12, max_iter=1000)
        cl = nx.closeness_centrality(G)
        bc = nx.betweenness_centrality(G, normalized=False)
        core = nx.core_number(nx.Graph(G))
        for i, v in enumerate(order):
            self.assertAlmostEqual(m["pagerank"][i], pr[v], places=8)
            self.assertAlmostEqual(m["closeness"][i], cl[v], places=8)
            self.assertAlmostEqual(m["betweenness"][i], bc[v], places=8)
            self.assertEqual(m["coreness"][i], core[v])

    def test_undirected(self):
        nx = self.nx
        g, edges, n = self._random_graph(False, 2)
        G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
        m, _ = compute_all(g)
        order = [int(l) for l in g.labels]
        bc = nx.betweenness_centrality(G, normalized=False)
        cl = nx.closeness_centrality(G)
        for i, v in enumerate(order):
            self.assertAlmostEqual(m["betweenness"][i], bc[v], places=8)
            self.assertAlmostEqual(m["closeness"][i], cl[v], places=8)

    def test_sampling_close_to_exact(self):
        g, _, _ = self._random_graph(True, 3)
        _, bc_exact = path_measures(g)
        _, bc_samp = path_measures(g, pivots=30, seed=1)
        rho = np.corrcoef(bc_exact, bc_samp)[0, 1]
        self.assertGreater(rho, 0.8)


class TestGraph(unittest.TestCase):
    def test_loops_and_parallel_edges(self):
        g = Graph.from_edges([("a", "a"), ("a", "b"), ("a", "b"), ("b", "a")])
        self.assertEqual(g.m, 2)
        self.assertEqual(g.A[0, 1], 2.0)         # параллельные рёбра складываются

    def test_undirected_pairs_not_doubled(self):
        g = Graph.from_edges([("a", "b"), ("b", "a")], directed=False)
        self.assertEqual(g.m, 1)
        self.assertEqual(g.A[0, 1], 1.0)

    def test_dangling_pagerank(self):
        g = Graph.from_edges([("a", "b")])        # b без исходящих рёбер
        pr = pagerank(g)
        self.assertAlmostEqual(float(pr.sum()), 1.0, places=9)
        self.assertGreater(pr[1], pr[0])


if __name__ == "__main__":
    unittest.main()

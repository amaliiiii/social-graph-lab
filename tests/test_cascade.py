"""Проверка симулятора каскадов (часть I, раздел 3; часть III, эксперимент 3)."""
import itertools
import unittest

import numpy as np

from sgl.cascade import Spread, exact_ic_spread, mean_ci, monte_carlo, simulate
from sgl.graph import Graph


def chain() -> Spread:
    # a -> b -> c в графе ВНИМАНИЯ: информация идёт от c к b и к a. Чтобы получить цепочку
    # «a влияет на b, b влияет на c», задаём рёбра внимания b -> a и c -> b.
    return Spread(Graph.from_edges([("b", "a"), ("c", "b")], nodes=["a", "b", "c"]))


class TestIC(unittest.TestCase):
    def test_chain_exact_value(self):
        # σ({a}) = 1 + 0,5 + 0,25 = 1,75 (часть I, раздел 3.2)
        self.assertAlmostEqual(exact_ic_spread(chain(), [0], "ic", 0.5), 1.75, places=12)

    def test_chain_monte_carlo(self):
        sizes = monte_carlo(chain(), [0], "ic", 0.5, runs=4000)
        m, ci = mean_ci(sizes)
        self.assertLess(abs(m - 1.75), max(3 * ci / 1.96, 0.05))

    def test_probability_one_reaches_everyone_who_can(self):
        g = Graph.from_edges([("b", "a"), ("c", "b"), ("d", "c"), ("x", "y")])
        sp = Spread(g)
        steps = simulate(sp, [g.labels.index("a")], "ic", 1.0)
        got = {g.labels[v] for st in steps for v in st}
        self.assertEqual(got, {"a", "b", "c", "d"})
        self.assertEqual([len(s) for s in steps], [1, 1, 1, 1])   # по одному шагу на звено

    def test_probability_zero_stays_at_seeds(self):
        sp = chain()
        self.assertEqual(simulate(sp, [0], "ic", 0.0), [[0]])

    def test_reproducible_and_common_worlds(self):
        sp = Spread(Graph.from_edges([(i, j) for i in range(20) for j in range(20)
                                      if i != j and (i * 7 + j * 3) % 5 == 0]))
        a = simulate(sp, [0], "ic", 0.3, run=5)
        b = simulate(sp, [0], "ic", 0.3, run=5)
        self.assertEqual(a, b)

    def test_monotone_and_submodular_exactly(self):
        """Свойства, на которых держится гарантия 1 − 1/e (часть I, раздел 3.5)."""
        rng = np.random.default_rng(4)
        pairs = [(int(a), int(b)) for a, b in rng.integers(0, 6, size=(40, 2)) if a != b]
        edges = sorted(set(pairs))[:12]
        sp = Spread(Graph.from_edges(edges, nodes=range(6)))
        f = lambda S: exact_ic_spread(sp, list(S), "ic", 0.4)
        nodes = range(sp.n)
        for r in range(0, 3):
            for S in itertools.combinations(nodes, r):
                for extra in nodes:
                    if extra in S:
                        continue
                    T = tuple(sorted(set(S) | {extra}))
                    self.assertGreaterEqual(f(T) + 1e-12, f(S))          # монотонность
                    for v in nodes:
                        if v in T:
                            continue
                        gain_S = f(S + (v,)) - f(S)
                        gain_T = f(T + (v,)) - f(T)
                        self.assertGreaterEqual(gain_S + 1e-12, gain_T)   # убывающая отдача


class TestWCandLT(unittest.TestCase):
    def test_weighted_cascade_probability_is_inverse_outdegree(self):
        # u подписан на двух авторов: каждый активирует u с вероятностью 1/2 в модели WC
        g = Graph.from_edges([("u", "a"), ("u", "b")])
        sp = Spread(g)
        a = g.labels.index("a")
        self.assertAlmostEqual(exact_ic_spread(sp, [a], "wc"), 1.5, places=12)

    def test_lt_threshold_needs_enough_neighbours(self):
        # u подписан на двух авторов; одного активного мало лишь при пороге > 1/2
        g = Graph.from_edges([("u", "a"), ("u", "b")])
        sp = Spread(g)
        a, u = g.labels.index("a"), g.labels.index("u")
        sizes = monte_carlo(sp, [a], "lt", runs=4000)
        self.assertAlmostEqual(sizes.mean(), 1.5, delta=0.05)      # P(θ ≤ 1/2) = 1/2
        both = monte_carlo(sp, [a, g.labels.index("b")], "lt", runs=50)
        self.assertTrue((both == 3).all())                         # порог ≤ 1 выполнен всегда

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            simulate(chain(), [0], "xx")


if __name__ == "__main__":
    unittest.main()

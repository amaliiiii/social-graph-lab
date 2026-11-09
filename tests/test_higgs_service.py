import gzip
import os
import tempfile
import unittest

import numpy as np

from sgl import higgs, service


def write_gz(path, lines):
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


class TestHiggsLoader(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        # 1 -> 2, 1 -> 3, 2 -> 3 ; 4 -> 3 (подписки: «A B» — A подписан на B)
        write_gz(os.path.join(self.dir, "social_network.edgelist.gz"), ["1 2", "1 3", "2 3", "4 3", "5 3"])
        # 1, 2, 4 репостили 3; 5 репостил 2 (вес — число репостов)
        write_gz(os.path.join(self.dir, "retweet_network.edgelist.gz"), ["1 3 2", "2 3 1", "4 3 5", "5 2 1"])
        write_gz(os.path.join(self.dir, "mention_network.edgelist.gz"), ["1 2 1"])

    def test_layers_share_vertex_set(self):
        layers = higgs.load_layers(self.dir)
        self.assertEqual({"social", "retweet", "mention", "union"}, set(layers))
        n = {g.n for g in layers.values()}
        self.assertEqual(n, {5})
        rt = layers["retweet"]
        i1, i3 = rt.labels.index(1), rt.labels.index(3)
        self.assertEqual(rt.A[i1, i3], 2.0)         # вес прочитан
        self.assertEqual(layers["social"].m, 5)

    def test_single_layer(self):
        g = higgs.load_layer(self.dir, "social")
        self.assertEqual((g.n, g.m), (5, 5))

    def test_direction_check(self):
        layers = higgs.load_layers(self.dir)
        ok = higgs.check_direction(layers["social"], layers["retweet"])
        flipped = higgs.load_layers(self.dir, reverse_social=True)
        bad = higgs.check_direction(flipped["social"], flipped["retweet"])
        self.assertGreater(ok["rho_in"], ok["rho_out"])
        self.assertEqual(ok["verdict"], "направление верное")
        self.assertGreater(bad["rho_out"], bad["rho_in"])
        self.assertIn("развернуть", bad["verdict"])

    def test_missing_dir(self):
        with self.assertRaises(FileNotFoundError):
            higgs.load_layers(tempfile.mkdtemp())


class TestService(unittest.TestCase):
    G2 = {"dataset": "demo-g2"}

    def test_analyze_demo(self):
        a = service.analyze(self.G2)
        self.assertEqual((a["n"], a["m"], a["shown"]), (5, 7, 5))
        self.assertEqual(len(a["edges"]), 7)
        node = {x["label"]: x["m"] for x in a["nodes"]}
        self.assertAlmostEqual(node["A"]["pagerank"], 0.282, delta=0.0006)

    def test_analyze_undirected_edges_once(self):
        a = service.analyze({"dataset": "demo-g1"})
        self.assertEqual(len(a["edges"]), 9)
        self.assertFalse(a["directed"])

    def test_upload_and_view_limit(self):
        text = "# сеть\n" + "\n".join(f"{i} {(i * 7 + 1) % 300} 1" for i in range(300))
        a = service.analyze({"edges_text": text, "directed": True}, max_nodes=50)
        self.assertEqual(a["shown"], 50)
        self.assertEqual(a["n"], 300)

    def test_bad_upload(self):
        with self.assertRaises(ValueError):
            service.analyze({"edges_text": "   \n# пусто"})
        with self.assertRaises(ValueError):
            service.analyze({"edges_text": "одно"})
        with self.assertRaises(ValueError):
            service.analyze({"dataset": "nope"})

    def test_cascade_follows_followers(self):
        # E подписан на D, D подписан на A: сообщение A доходит до B, C, D, затем до E
        c = service.cascade(self.G2, ["A"], "ic", 1.0)
        self.assertEqual([sorted(s["nodes"]) for s in c["steps"]], [["A"], ["B", "C", "D"], ["E"]])
        self.assertEqual(c["total"], 5)

    def test_cascade_errors(self):
        with self.assertRaises(ValueError):
            service.cascade(self.G2, [], "ic", 0.5)
        with self.assertRaises(ValueError):
            service.cascade(self.G2, ["Z"], "ic", 0.5)
        with self.assertRaises(ValueError):
            service.cascade(self.G2, ["A"], "ic", 1.5)

    def test_compare_top_beats_random_on_social(self):
        r = service.compare({"dataset": "demo-social"}, "pagerank", 5, "ic", 0.1, runs=80)
        self.assertGreater(r["top"]["mean"], r["random"]["mean"])
        self.assertGreater(r["gain"]["mean"], r["gain"]["ci"])    # разница значима
        self.assertEqual(len(r["seeds"]), 5)

    def test_quiz(self):
        g1 = {"dataset": "demo-g1"}
        self.assertTrue(service.quiz(g1, "c", "betweenness", 14)["correct"])
        r = service.quiz(g1, "c", "betweenness", 13)
        self.assertFalse(r["correct"])
        self.assertEqual(r["value"], 14.0)
        self.assertTrue(service.quiz(g1, "c", "closeness", 0.58)["correct"])


if __name__ == "__main__":
    unittest.main()

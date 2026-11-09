import unittest

from sgl.extract import build_layers, evaluate, extract_all, extract_message


def triples(items):
    return {(i.source, i.target, i.kind) for i in items}


class TestExtract(unittest.TestCase):
    def test_repost_and_mentions(self):
        r = extract_message("Ann", "RT @Bob: привет @carol и ещё раз @carol")
        self.assertEqual(triples(r), {("ann", "bob", "repost"), ("ann", "carol", "mention")})

    def test_email_is_not_mention(self):
        self.assertEqual(extract_message("ann", "пишите на info@site.ru"), [])

    def test_reply_not_double_counted(self):
        r = extract_message("ann", "@bob согласна, ещё @dan", reply_to="bob")
        self.assertEqual(triples(r), {("ann", "bob", "reply"), ("ann", "dan", "mention")})

    def test_self_references_removed(self):
        self.assertEqual(extract_message("ann", "RT @ann: я и @ann"), [])

    def test_rt_only_at_start(self):
        r = extract_message("ann", "смотрите RT @bob: так нельзя")
        self.assertEqual(triples(r), {("ann", "bob", "mention")})

    def test_layers_weights(self):
        msgs = [{"author": "a", "text": "@b @c"}, {"author": "a", "text": "ещё @b"},
                {"author": "c", "text": "RT @a: x"}]
        layers = build_layers(extract_all(msgs))
        m = layers["mention"]
        ia, ib = m.labels.index("a"), m.labels.index("b")
        self.assertEqual(m.A[ia, ib], 2.0)
        self.assertEqual(layers["repost"].m, 1)

    def test_evaluate(self):
        r = evaluate({1, 2, 3}, {2, 3, 4})
        self.assertAlmostEqual(r["precision"], 2 / 3)
        self.assertAlmostEqual(r["recall"], 2 / 3)


if __name__ == "__main__":
    unittest.main()

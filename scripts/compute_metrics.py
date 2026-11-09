"""Этап 4: таблица «вершина × мера» для слоя набора Higgs.

Примеры:
  python scripts/compute_metrics.py --data ./data --layer retweet --out retweet_metrics.csv
  python scripts/compute_metrics.py --data ./data --layer social --skip-paths --check-direction

Близость и посредничество считаются на Python по выборке из --pivots начальных вершин
(полный граф подписок: 15 млн рёбер, один обход занимает десятки секунд), поэтому для
полного графа подписок их обычно пропускают (--skip-paths) или берут малое число начальных
вершин и запускают расчёт на ночь.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgl import higgs                      # noqa: E402
from sgl.measures import compute_all       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="каталог с файлами набора Higgs")
    ap.add_argument("--layer", default="social", choices=["social", "retweet", "reply", "mention", "union"])
    ap.add_argument("--out", default="metrics.csv")
    ap.add_argument("--pivots", type=int, default=100, help="число начальных вершин для оценки пути")
    ap.add_argument("--exact-limit", type=int, default=400)
    ap.add_argument("--skip-paths", action="store_true", help="не считать близость и посредничество")
    ap.add_argument("--reverse-social", action="store_true", help="развернуть рёбра графа подписок")
    ap.add_argument("--check-direction", action="store_true", help="проверить направление графа подписок")
    args = ap.parse_args()

    t = time.time()
    layers = higgs.load_layers(args.data, reverse_social=args.reverse_social)
    print(f"загружено слоёв: {sorted(layers)}; вершин: {next(iter(layers.values())).n}; "
          f"{time.time() - t:.1f} с")
    for name, g in layers.items():
        print(f"  {name:8s} рёбер: {g.m}")
    if args.check_direction:
        if "social" not in layers or "retweet" not in layers:
            sys.exit("для проверки нужны слои social и retweet")
        print("проверка направления:", higgs.check_direction(layers["social"], layers["retweet"]))

    g = layers[args.layer]
    t = time.time()
    metrics, modes = compute_all(g, exact_limit=args.exact_limit, pivots=args.pivots,
                                 skip_paths=args.skip_paths)
    print(f"меры вычислены за {time.time() - t:.1f} с; режим близости и посредничества: {modes['betweenness']}")

    names = ["in_degree", "out_degree", "closeness", "betweenness", "eigenvector",
             "pagerank", "hits_authority", "hits_hub", "coreness"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id"] + names)
        for i in range(g.n):
            w.writerow([g.label(i)] + [f"{metrics[k][i]:.8g}" for k in names])
    print("записано:", args.out)


if __name__ == "__main__":
    main()

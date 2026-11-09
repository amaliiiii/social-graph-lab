"""Загрузка набора Higgs Twitter (SNAP) и построение графов.

Ожидаемые файлы в каталоге данных (имена как в коллекции SNAP):
  social_network.edgelist.gz    подписки,   строки «A B»
  retweet_network.edgelist.gz   репосты,    строки «A B вес»
  reply_network.edgelist.gz     ответы,     строки «A B вес»
  mention_network.edgelist.gz   упоминания, строки «A B вес»
  higgs-activity_time.txt.gz    журнал,     строки «A B время тип»

Формат строк проверен только по описанию набора: если в ваших файлах иначе,
правьте `read_edgelist`. Направление ребра в графе подписок нужно проверить
функцией `check_direction` (см. часть III, раздел 2, шаг 3).
"""
from __future__ import annotations

import os

import numpy as np

from .graph import Graph

LAYER_FILES = {
    "social": ("social_network.edgelist.gz", False),
    "retweet": ("retweet_network.edgelist.gz", True),
    "reply": ("reply_network.edgelist.gz", True),
    "mention": ("mention_network.edgelist.gz", True),
}


def read_edgelist(path: str, weighted: bool = False):
    """Читает список рёбер (в том числе .gz): возвращает исходные номера src, dst и веса."""
    import pandas as pd  # импорт здесь, чтобы остальной пакет не требовал pandas

    cols = [0, 1, 2] if weighted else [0, 1]
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", usecols=cols,
                     dtype={0: np.int64, 1: np.int64}, engine="c")
    src, dst = df[0].to_numpy(), df[1].to_numpy()
    w = df[2].to_numpy(dtype=float) if weighted and df.shape[1] > 2 else np.ones(len(src))
    return src, dst, w


def available_layers(data_dir: str) -> list[str]:
    return [name for name, (fn, _) in LAYER_FILES.items()
            if os.path.exists(os.path.join(data_dir, fn))]


def load_layer(data_dir: str, name: str, reverse: bool = False) -> Graph:
    """Один слой на собственном множестве вершин (метки — исходные номера)."""
    fn, weighted = LAYER_FILES[name]
    src, dst, w = read_edgelist(os.path.join(data_dir, fn), weighted)
    if reverse:
        src, dst = dst, src
    ids, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    k = len(src)
    return Graph(len(ids), inv[:k], inv[k:], w, labels=ids.tolist())


def load_layers(data_dir: str, reverse_social: bool = False) -> dict[str, Graph]:
    """Все слои на общем множестве вершин (индексы согласованы) + объединённый граф."""
    raw = {}
    for name in available_layers(data_dir):
        fn, weighted = LAYER_FILES[name]
        raw[name] = read_edgelist(os.path.join(data_dir, fn), weighted)
    if not raw:
        raise FileNotFoundError(f"в {data_dir} нет файлов набора Higgs")
    ids = np.unique(np.concatenate([np.concatenate([s, d]) for s, d, _ in raw.values()]))
    layers = {}
    for name, (s, d, w) in raw.items():
        if name == "social" and reverse_social:
            s, d = d, s
        layers[name] = Graph(len(ids), np.searchsorted(ids, s), np.searchsorted(ids, d),
                             w, labels=ids.tolist())
    union = None
    for g in layers.values():
        union = g if union is None else union.union(g)
    layers["union"] = union
    return layers


def check_direction(social: Graph, retweet: Graph) -> dict:
    """Проверка направления рёбер в графе подписок.

    Если ребро читается как «u подписан на v», то входящая степень v в графе подписок
    (число подписчиков) должна быть сильнее связана с числом разных пользователей,
    репостнувших v, чем исходящая. Графы заданы на одном множестве вершин."""
    from scipy.stats import spearmanr

    from .measures import in_degree, out_degree

    if social.n != retweet.n:
        raise ValueError("нужны слои на общем множестве вершин (load_layers)")
    r = in_degree(retweet)
    mask = (r > 0) | (in_degree(social) > 0) | (out_degree(social) > 0)
    rho_in = float(spearmanr(in_degree(social)[mask], r[mask])[0])
    rho_out = float(spearmanr(out_degree(social)[mask], r[mask])[0])
    return {"rho_in": rho_in, "rho_out": rho_out,
            "verdict": "направление верное" if rho_in >= rho_out else "граф подписок нужно развернуть"}

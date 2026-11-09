"""Этап 2: извлечение связей из текста сообщений.

Правила (часть II доклада):
  * «RT @имя: ...» в начале сообщения   -> связь «репост» автор -> имя;
  * @имя в тексте                       -> связь «упоминание»; упоминание того, кого
                                           репостят или кому отвечают, отдельно не считается
                                           (оно уже выражено связью «репост» / «ответ»);
  * поле reply_to у сообщения            -> связь «ответ» автор -> адресат.
Повторные упоминания одного имени в сообщении считаются один раз, самоссылки удаляются,
имена приводятся к нижнему регистру. Адреса почты (a@b.ru) упоминаниями не считаются.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .graph import Graph

HANDLE = r"\w{1,32}"
MENTION_RE = re.compile(rf"(?<![\w@])@({HANDLE})")
RT_RE = re.compile(rf"^\s*RT\s+@({HANDLE})\s*:", re.IGNORECASE)


@dataclass(frozen=True)
class Interaction:
    source: str
    target: str
    kind: str          # 'repost' | 'mention' | 'reply'
    time: float | None = None


def extract_message(author: str, text: str, reply_to: str | None = None,
                    time: float | None = None) -> list[Interaction]:
    a = author.strip().lstrip("@").lower()
    out: list[Interaction] = []
    exclude: set[str] = set()

    m = RT_RE.match(text)
    if m:
        t = m.group(1).lower()
        exclude.add(t)
        if t != a:
            out.append(Interaction(a, t, "repost", time))
    if reply_to:
        t = reply_to.strip().lstrip("@").lower()
        exclude.add(t)
        if t != a:
            out.append(Interaction(a, t, "reply", time))

    seen: set[str] = set()
    for mm in MENTION_RE.finditer(text):
        t = mm.group(1).lower()
        if t in seen or t in exclude or t == a:
            continue
        seen.add(t)
        out.append(Interaction(a, t, "mention", time))
    return out


def extract_all(messages) -> list[Interaction]:
    """messages — словари с ключами author, text и (необязательно) reply_to, time."""
    res: list[Interaction] = []
    for msg in messages:
        res.extend(extract_message(msg["author"], msg["text"],
                                   msg.get("reply_to"), msg.get("time")))
    return res


def build_layers(interactions: list[Interaction]) -> dict[str, Graph]:
    """Три взвешенных графа (вес = число повторений) на общем множестве вершин."""
    users = list(dict.fromkeys(x for i in interactions for x in (i.source, i.target)))
    layers = {}
    for kind in ("repost", "mention", "reply"):
        edges = [(i.source, i.target) for i in interactions if i.kind == kind]
        layers[kind] = Graph.from_edges(edges, directed=True, nodes=users)
    return layers


def evaluate(predicted: set, gold: set) -> dict:
    """Точность и полнота извлечения на размеченном вручную корпусе."""
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}

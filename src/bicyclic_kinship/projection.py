"""The terminal class lane: `cls(a o b) = cls(b)`.

Some distinctions are carried by the LAST step of a chain and by nothing else. Gender is
one: whoever a long chain passes through, `... o mother` names a woman. So is the in-law
boundary at the point where it is terminal. Such a law is a monoid action rather than a
group one -- it is not invertible, and it composes by taking the right operand's class.

The derivation is a union-find, not a search: every triple `a o b = c` forces `c ~ b`, and
the finest partition satisfying the law is the transitive closure of those forcings. A
single block means the algebra admits no such axis and the lane is dropped.

The mirror law `cls(a o b) = cls(a)` is the same construction on the other side; it is
derived too and survives only where it holds. On CLUTRR the right lane yields six classes
and the left lane collapses to one and is dropped -- neither outcome is declared.
"""
from __future__ import annotations

from .derive import Triple

Partition = dict[str, str]


def projection_axis(triples: list[Triple], side: str = "right") -> Partition:
    """The finest partition with `cls(a o b) = cls(b)` (or `cls(a)` for side='left')."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    syms = {s for t in triples for s in t}
    parent = {s: s for s in syms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in triples:
        ra, rb = find(c), find(b if side == "right" else a)
        if ra != rb:
            parent[ra] = rb
    return {s: find(s) for s in syms}


def n_classes(axis: Partition) -> int:
    return len(set(axis.values()))

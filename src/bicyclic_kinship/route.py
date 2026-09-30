"""The route lane: a finite-state memory of the path a chain took. OPT-IN, experimental.

Every lane in `system.py` is a function of where a chain LANDS. Some distinctions are a
property of the route instead: two symbols can sit at the same address and differ only in
what happened on the way there. No homomorphism into the existing families can separate
them, because they are the same point.

This lane gives each symbol a small derived LABEL and folds a chain through a derived
transition relation:

    state(c1)                 = label(c1)
    state(c1 .. ck, b)        = delta(state(c1 .. ck), key(b))
    a symbol s is admitted    iff label(s) is among the states the chain can be in

`key(b)` is what the transition reads about the next step. Two readings are supported:

    "move"         the step's DERIVED coordinates (additive and cone) and nothing else
    "move+label"   those coordinates plus the step's own derived label

DERIVATION escalates the number of states from one and stops at the least that works. A
labelling works when the transition relation it induces is no less deterministic than the
observed product table itself (the finest labelling, every symbol its own state). The
search is exact backtracking with a node budget. Two constraints shape it:

    SEPARATION   symbols the other lanes cannot tell apart (their `collisions`) must keep
                 distinct labels. Those are the only distinctions this lane exists for, and
                 if there are none the lane derives nothing.
    COMPRESSION  the result must have fewer states than there are symbols, and no more
                 than `bound`. An automaton with a state per symbol is the composition
                 table itself, not a law about it.

Exact minimisation is NP-hard in general; here the separation constraint and the small
bound keep it tractable, and a search that exceeds its budget derives nothing rather than
returning a larger automaton.

SOUNDNESS. The transition relation is set-valued: a product the data answers two ways
leads to both states, and a transition never observed ABSTAINS rather than guessing. The
lane is then audited like every other law -- it must admit the known answer on every
training chain of every length -- and is dropped entirely if a single chain contradicts it.
It can only narrow what the other lanes admit.
"""
from __future__ import annotations

from collections import defaultdict

from .system import AxisSystem, collisions

Chain = tuple[str, ...]


class RouteLaw:
    """A derived labelling, a transition relation, and the fold they define."""

    def __init__(self, label: dict[str, int], delta: dict[tuple, frozenset[int]],
                 keys: dict[str, tuple]):
        self.label = label
        self.delta = delta
        self.keys = keys

    @property
    def n_states(self) -> int:
        return len(set(self.label.values()))

    def fold(self, chain) -> frozenset[int] | None:
        """The states the chain can end in, or None when some step was never observed --
        which means abstain, not reject."""
        if not chain or any(r not in self.label for r in chain):
            return None
        states = frozenset({self.label[chain[0]]})
        for r in chain[1:]:
            nxt: set[int] = set()
            for q in states:
                got = self.delta.get((q, self.keys[r]))
                if got is None:
                    return None
                nxt |= got
            states = frozenset(nxt)
        return states

    def admits(self, chain, symbol: str) -> bool:
        states = self.fold(tuple(chain))
        return states is None or self.label.get(symbol) in states

    def describe(self) -> str:
        return f"route lane: {self.n_states} states, {len(self.delta)} transitions"


def _geometry(system: AxisSystem, s: str) -> tuple:
    """A symbol's derived coordinates on the lanes that MOVE: additive and cone."""
    return (tuple(g[s] for g in system.additive), tuple(c[s] for c in system.cone))


def _relation(products, label, keys):
    """Build the set-valued transition relation, and count keys whose products disagree."""
    by_key: dict[tuple, set[frozenset[int]]] = defaultdict(set)
    for (a, b), answers in products.items():
        by_key[(label[a], keys[b])].add(frozenset(label[c] for c in answers))
    delta = {k: frozenset().union(*v) for k, v in by_key.items()}
    conflicts = sum(1 for v in by_key.values() if len(v) > 1)
    return delta, conflicts


def derive_route(pairs, system: AxisSystem, read: str = "move+label",
                 bound: int = 4, budget: int = 2_000_000) -> RouteLaw | None:
    """The smallest route automaton: escalate the number of states from one, stop at the
    least that fits, and return None when there is nothing to separate, nothing fits
    within `bound` (or within the search `budget`), or the audit retires the result."""
    if read not in ("move", "move+label"):
        raise ValueError("read must be 'move' or 'move+label'")
    syms = sorted(system.symbols)
    known = set(syms)
    groups = collisions({s: system.coordinates(s) for s in syms})
    if not groups:
        return None
    geom = {s: _geometry(system, s) for s in syms}

    products: dict[tuple[str, str], set[str]] = defaultdict(set)
    for chain, answer in pairs:
        if len(chain) == 2 and answer in known and all(r in known for r in chain):
            products[tuple(chain)].add(answer)
    prods = [(a, b, frozenset(cs)) for (a, b), cs in products.items()]

    def key(lab, a, b):
        return (lab[a], geom[b]) if read == "move" else (lab[a], geom[b], lab[b])

    # The finest labelling is the product table itself; whatever nondeterminism it has is
    # the data's own, and a coarser labelling may not add to it.
    finest = {s: i for i, s in enumerate(syms)}
    allowed = _relation(products, finest,
                        {s: (geom[s],) if read == "move" else (geom[s], finest[s])
                         for s in syms})[1]

    # Collision partners first, so the separation constraint bites early.
    order = [x for g in groups for x in g]
    order += [x for x in syms if x not in set(order)]
    partner = {a: {b for g in groups if a in g for b in g if b != a} for a in syms}
    touching = defaultdict(list)
    for i, (a, b, cs) in enumerate(prods):
        last = max((order.index(x) for x in (a, b, *cs)))
        touching[order[last]].append(i)

    def search(n):
        lab: dict[str, int] = {}
        table: dict[tuple, list[frozenset[int]]] = defaultdict(list)
        nodes = [0]

        def go(i, used, bad):
            nodes[0] += 1
            if nodes[0] > budget:
                raise TimeoutError
            if i == len(order):
                return dict(lab)
            s = order[i]
            for q in range(min(used + 1, n)):
                if any(lab.get(p) == q for p in partner[s]):
                    continue
                lab[s] = q
                added, extra = [], 0
                for j in touching[s]:
                    a, b, cs = prods[j]
                    k = key(lab, a, b)
                    t = frozenset(lab[c] for c in cs)
                    if table[k] and t not in table[k]:
                        extra += 1
                    table[k].append(t)
                    added.append(k)
                if bad + extra <= allowed:
                    got = go(i + 1, max(used, q + 1), bad + extra)
                    if got is not None:
                        return got
                for k in added:
                    table[k].pop()
                del lab[s]
            return None

        return go(0, 0, 0)

    label = None
    try:
        for n in range(1, min(bound, len(syms) - 1) + 1):
            label = search(n)
            if label is not None:
                break
    except TimeoutError:
        return None
    if label is None:
        return None
    keys = {s: (geom[s],) if read == "move" else (geom[s], label[s]) for s in syms}
    delta, _ = _relation(products, label, keys)
    law = RouteLaw(label, delta, keys)

    for chain, answer in pairs:                        # the audit, at every length
        chain = tuple(chain)
        if answer in known and all(r in known for r in chain) and not law.admits(chain, answer):
            return None
    return law

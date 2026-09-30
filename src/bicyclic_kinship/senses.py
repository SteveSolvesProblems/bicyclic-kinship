"""Senses: one surface symbol that stands for several points. Opt-in; nothing uses it by
default.

Every lane in `system.py` gives each symbol ONE coordinate. A vocabulary can break that
without being wrong: a word may name two different places, the way one word can cover a
relation in either direction. Then no single coordinate satisfies all of that word's
products, the lanes that need one are lost for the WHOLE vocabulary, and answers widen to
near-abstention. Sound, and useless.

The repair is to let a symbol split into senses, each sense an ordinary symbol at its own
address, and to derive the lanes over the senses. Which symbols split, and into how many,
is read off the data:

    1. Take the symbols that only ever appear as the ANSWER of a product (`terminal`).
       Everything else -- the operands -- is derived first, from the products whose answer
       is itself an operand (`core`). A terminal symbol constrains nothing but itself, so
       leaving it out of that first pass costs nothing it could have supplied honestly.
    2. Each product `a o b = w` with `w` terminal then IMPLIES where that occurrence of `w`
       sits, lane by lane: `g(a) + g(b)`, the cone fold of `a` and `b`, the class of `b`,
       the class of `a`. Occurrences that imply the same point are one sense; occurrences
       that imply different points are different senses. The number of senses is the
       number of distinct points, which is the fewest any assignment could use.
    3. Occam, checked rather than assumed: a split survives only if merging it back makes
       the re-derived structure strictly weaker -- fewer additive axes, no cone, or coarser
       classes over the operands. A split that buys nothing is undone.

Answers are read back through the senses: a chain's admitted senses are mapped to their
surface words. A chain whose OPERANDS carry senses is composed once per combination and the
answers are united, which only ever widens the set, so it cannot cost soundness.

SCOPE, stated rather than hidden: only terminal symbols are split. A symbol that is both an
operand and polysemous produces conflicting products, which `binary_table` already drops;
recovering senses from those is not attempted here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product

from .cone import cone_compose
from .derive import Triple, triples_from
from .system import AxisSystem

SEP = "′"          # sense k of `w` is `w` followed by k primes; never a surface name


def _restricted_classes(part: dict | None, keep: set[str]) -> int:
    if part is None:
        return 1
    return len({part[s] for s in keep if s in part})


def structure(system: AxisSystem, operands: set[str]) -> tuple:
    """How much a derivation found, measured on the operands only -- the symbols a split
    never renames, so a split and its merge are compared on the same ground."""
    return (len(system.additive), bool(system.cone),
            _restricted_classes(system.right, operands),
            _restricted_classes(system.left, operands))


def _not_weaker(a: tuple, b: tuple) -> bool:
    return all(x >= y for x, y in zip(a, b))


def _implied(core: AxisSystem, a: str, b: str) -> tuple:
    """Where `a o b` lands, lane by lane; None where a lane has no coordinate for it."""
    add = tuple(g[a] + g[b] if a in g and b in g else None for g in core.additive)
    cone = tuple(cone_compose(c[a], c[b]) if a in c and b in c else None for c in core.cone)
    r = core.right.get(b) if core.right is not None else None
    l = core.left.get(a) if core.left is not None else None
    return add + cone + (r, l)


def _group(keys: dict[tuple, list]) -> list[list]:
    """Occurrences by implied point. A key with unknown (None) components joins the one
    complete key it agrees with, when there is exactly one; otherwise it stands alone."""
    complete = [k for k in keys if None not in k]
    groups = {k: list(keys[k]) for k in complete}
    for k in keys:
        if None in k:
            fits = [c for c in complete
                    if all(x is None or x == y for x, y in zip(k, c))]
            if len(fits) == 1:
                groups[fits[0]].extend(keys[k])
            else:
                groups[k] = list(keys[k])
    return [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: sorted(kv[1]))]


class Senses:
    """The derived split: which surface symbols carry several senses, and which sense each
    observed product used."""

    def __init__(self, split: dict[str, list[list[tuple[str, str]]]]):
        #: surface symbol -> list of senses, each the operand pairs that produced it
        self.split = split
        self.sense_of: dict[tuple[str, str, str], str] = {}
        self.surface: dict[str, str] = {}
        for w, groups in split.items():
            for k, occ in enumerate(groups):
                name = w + SEP * (k + 1)
                self.surface[name] = w
                for a, b in occ:
                    self.sense_of[(a, b, w)] = name

    def senses(self, w: str) -> list[str]:
        if w not in self.split:
            return [w]
        return [w + SEP * (k + 1) for k in range(len(self.split[w]))]

    def to_surface(self, s: str) -> str:
        return self.surface.get(s, s)

    def relabel(self, triples: list[Triple]) -> list[Triple]:
        return [(a, b, self.sense_of.get((a, b, c), c)) for a, b, c in triples]

    def __len__(self) -> int:
        return len(self.split)


def derive_senses(triples: list[Triple], families=AxisSystem.FAMILIES,
                  bound: int = 4) -> Senses:
    """The fewest senses that let the lanes hold. Empty when nothing needs splitting.

    `bound` caps the senses per symbol: a symbol whose occurrences imply more distinct
    points than that is left whole, since at that point the word is closer to "anything"
    than to a few meanings."""
    operands = {a for a, _, _ in triples} | {b for _, b, _ in triples}
    terminal = {c for _, _, c in triples} - operands
    core = [t for t in triples if t[2] in operands]
    if not terminal or not core:
        return Senses({})
    first = AxisSystem.discover(core, families)

    keys: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    for a, b, c in triples:
        if c in terminal:
            keys[c][_implied(first, a, b)].append((a, b))
    candidates = {}
    for w, ks in keys.items():
        groups = _group(ks)
        if 1 < len(groups) <= bound:
            candidates[w] = groups
    if not candidates:
        return Senses({})

    def derive(split):
        return AxisSystem.discover(Senses(split).relabel(triples), families)

    whole = structure(AxisSystem.discover(triples, families), operands)
    best = structure(derive(candidates), operands)
    if _not_weaker(whole, best):
        return Senses({})                 # splitting buys nothing: keep every word whole
    for w in sorted(candidates):          # undo every split the structure does not need
        trial = {k: v for k, v in candidates.items() if k != w}
        if _not_weaker(structure(derive(trial), operands), best):
            candidates = trial
    return Senses(candidates)


class SensedSolver:
    """A `Solver` over senses, answering in surface words. Opt-in.

    Built only by `train`. With no split found it is exactly the plain `Solver`."""

    #: the most sense combinations a single chain is expanded into before abstaining
    MAX_EXPANSIONS = 64

    def __init__(self, inner, senses: Senses, surface_vocab: list[str], unresolved: int):
        self.inner = inner
        self.senses = senses
        self.vocab = sorted(surface_vocab)
        #: training chains whose answer no sense of it could be matched to by the laws
        self.unresolved = unresolved

    @classmethod
    def train(cls, pairs, families=AxisSystem.FAMILIES, bound: int = 4, **kw):
        from .solver import Solver
        pairs = [(tuple(c), a) for c, a in pairs if c]
        vocab = sorted({s for c, a in pairs for s in (*c, a)})
        senses = derive_senses(triples_from(pairs), families, bound)
        if not senses:
            return cls(Solver.train(pairs, families=families, **kw), senses, vocab, 0)
        # Relabel every training answer with the sense it used. Products carry theirs
        # from the derivation; a longer chain takes the sense the laws admit there. One
        # that no sense fits keeps the first, so the audit sees the contradiction and
        # retires whichever lane it falsifies -- the audit is never skipped.
        system = AxisSystem.discover(senses.relabel(triples_from(pairs)), families)
        out, unresolved = [], 0
        for c, a in pairs:
            if a in senses.split:
                key = (c[0], c[1], a) if len(c) == 2 else None
                s = senses.sense_of.get(key) if key else None
                if s is None:
                    admitted = set()
                    for e in cls._expand(senses, c):
                        admitted.update(system.compose(list(e)))
                    fits = [x for x in senses.senses(a) if x in admitted]
                    s = fits[0] if fits else senses.senses(a)[0]
                    unresolved += not fits
                a = s
            out.append((c, a))
        return cls(Solver.train(out, families=families, **kw), senses, vocab, unresolved)

    @staticmethod
    def _expand(senses: Senses, chain):
        options = [senses.senses(r) for r in chain]
        n = 1
        for o in options:
            n *= len(o)
        if n > SensedSolver.MAX_EXPANSIONS:
            return None
        return list(product(*options))

    @property
    def system(self) -> AxisSystem:
        return self.inner.system

    def known(self, chain) -> bool:
        return bool(chain) and all(r in set(self.vocab) for r in chain)

    def solve(self, chain) -> list[str]:
        chain = tuple(chain)
        if not self.known(chain):
            return list(self.vocab)
        exp = self._expand(self.senses, chain)
        if exp is None:
            return list(self.vocab)                     # too many readings: abstain
        out = set()
        for e in exp:
            out.update(self.senses.to_surface(n) for n in self.inner.solve(e))
        return sorted(out)

    def best(self, chain) -> str | None:
        names = self.solve(chain)
        if len(names) <= 1:
            return names[0] if names else None
        votes = Counter()
        for e in self._expand(self.senses, tuple(chain)) or []:
            b = self.inner.best(e)
            if b is not None:
                votes[self.senses.to_surface(b)] += 1
        picks = [n for n in names if votes[n] == max(votes.values(), default=0)]
        return min(picks) if picks else min(names)

    def describe(self) -> str:
        parts = [f"{w}: {len(g)} senses" for w, g in sorted(self.senses.split.items())]
        return f"{self.inner.system.describe()}; split {len(self.senses)} symbols" + (
            f" ({'; '.join(parts)})" if parts else "")

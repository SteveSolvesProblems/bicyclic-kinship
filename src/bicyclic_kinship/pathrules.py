"""Path rules: local, situational if-thens read off the chain itself.

The laws in `system.py` are GLOBAL: each is a property of the composite, recoverable by
folding. Where two symbols sit at the same address, no such law can separate them -- they
are terminal, they never appear as an operand in a distinguishing product, and searching
harder for a homomorphism does not help.

What separates them is a property of the PATH, not of the answer:

    if the chain crosses a spouse boundary and then rises, it is the in-law side.

That feature lives on the chain. It is read by scanning left to right once, it never
consults an intermediate result, and it never requires the answer symbol to appear as an
operand anywhere -- so terminality simply does not apply to it.

TWO CHANNELS, COMBINED, NOT MERGED:

    algebra  ->  the SOUND candidate set (every derived law must hold)
    path     ->  selects WITHIN that set

Soundness is preserved by construction: a path rule can only narrow a set the algebra
already admits, never add to it. When no rule applies, nothing is selected and the
ambiguity survives -- a gap is cheaper than an error.

DERIVED, NOT WRITTEN. A rule is computed from labelled chains in closed form -- no search
over the subset lattice, no hand-written condition. For a feature that reads some part of
the chain, the separating set S must be DISJOINT from that part of every x-chain and must
HIT that part of every y-chain: the first condition confines S, the second is a hitting-set
check. If no S exists for any feature kind, the pair is reported unseparable rather than
approximated.

THE FEATURE LANGUAGE is deliberately tiny -- `first`, `last`, `any`, plus one stateful
latch. Pure membership ("contains a spouse") derives nothing, since `husband o daughter =
daughter` traverses one and is not an in-law; the distinguishing feature is positional and
has state, which is why the language needs more than membership.

WHAT IT COSTS TO GET IT WRONG. Two guards here are not decoration. Sequential covering
without a support floor memorises: eleven rules for one collision class, zero training
residue, and held-out soundness of 83% with 189 genuine errors. And purity alone is not
enough -- a rule pure on training but covering only half of its own class produced 25
held-out errors, so a one-sided rule must also cover most of the class it claims.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

Chain = tuple[str, ...]


def chains_by_target(pairs: list[tuple[Chain, str]]) -> dict[str, list[Chain]]:
    out: dict[str, list[Chain]] = {}
    for chain, target in pairs:
        out.setdefault(target, []).append(tuple(chain))
    return out


def latch_value(chain: Chain, trigger: frozenset[str], gen: dict) -> str:
    """State after scanning: what happens on the first displacement move AFTER a trigger.

    Returns "+" (the first non-zero move after the trigger goes up), "-" (it goes down), or
    "none" (no trigger occurred). One left-to-right pass, no intermediate result consulted,
    decided before the answer is known.

    This is the smallest feature with STATE, and state is what the static tests lack: a
    static `first(chain)` test cannot express set-then-clear, which is exactly what is
    needed when a descending step ABSORBS a marital crossing (`husband o son = son`) while
    an ascending one preserves it (`husband o father = father-in-law`).

    Note what the readout reads: the sign of the DERIVED additive coordinate, not the
    identity of a symbol. A rule built on a coordinate the manifold already validated
    inherits that validation and extends to any symbol later placed at the same coordinate.
    """
    seen = False
    for r in chain:
        if seen and gen.get(r, 0) != 0:
            return "+" if gen[r] > 0 else "-"
        if r in trigger:
            seen = True
    return "none"


def derive_latch(xs: list[Chain], ys: list[Chain], gen: dict, vocab,
                 min_support: int = 1):
    """Smallest trigger set whose latch value separates x-chains from y-chains exactly.

    Enumerates singletons then pairs -- 210 candidates over a 20-symbol vocabulary -- and
    requires the value sets to be disjoint. No search over the whole subset lattice."""
    if min(len(xs), len(ys)) < min_support:
        return None
    for size in (1, 2):
        for T in combinations(sorted(vocab), size):
            T = frozenset(T)
            vx = {latch_value(c, T, gen) for c in xs}
            vy = {latch_value(c, T, gen) for c in ys}
            if vx and vy and not (vx & vy):
                return T, frozenset(vy)
    return None


def derive_onesided_latch(xs: list[Chain], ys: list[Chain], gen: dict, vocab,
                          min_support: int = 1, min_coverage: float = 0.9):
    """A latch value that occurs ONLY among y-chains. Fires -> y; otherwise ABSTAIN.

    A two-sided rule must partition the classes and is refused when it cannot. A one-sided
    rule is weaker and always available: it claims only the region where the evidence is
    unanimous and leaves the rest as a gap, which is how the model holds two possibilities
    at once.

    `min_coverage` is the guard against a rule that is pure by accident. A feature that
    misses half the class it claims to define is not capturing the concept -- and
    incompleteness on the positive side predicts unreliability on the negative side."""
    best = None
    for size in (1, 2):
        for T in combinations(sorted(vocab), size):
            T = frozenset(T)
            seen_x = {latch_value(c, T, gen) for c in xs}
            counts = Counter(latch_value(c, T, gen) for c in ys)
            pure = {v for v in counts if v not in seen_x}
            cover = sum(counts[v] for v in pure)
            if (cover >= min_support and ys and cover >= min_coverage * len(ys)
                    and (best is None or cover > best[2])):
                best = (T, frozenset(pure), cover)
    return best


#: Each kind reads one part of the chain. All are a single scan, none needs an intermediate.
KINDS = {
    "first": lambda chain: {chain[0]},
    "last": lambda chain: {chain[-1]},
    "any": lambda chain: set(chain),
}


def discriminating_set(xs: list[Chain], ys: list[Chain], kind: str) -> frozenset[str] | None:
    """Symbols that, read by `kind`, mark a y-chain and never an x-chain."""
    read = KINDS[kind]
    if not xs or not ys:
        return None
    forbidden = {s for c in xs for s in read(c)}
    universe = {s for c in ys for s in read(c)} - forbidden
    remaining = [read(c) & universe for c in ys]
    if any(not r for r in remaining):
        return None                      # some y-chain has nothing to mark it
    chosen: set[str] = set()
    while remaining:
        # sorted, so a tie between equally good symbols breaks the same way on every run
        best = max(sorted(universe - chosen), key=lambda s: sum(1 for r in remaining if s in r))
        if not any(best in r for r in remaining):
            return None
        chosen.add(best)
        remaining = [r for r in remaining if best not in r]
    return frozenset(chosen)


def derive_rule(xs: list[Chain], ys: list[Chain]):
    """The simplest single feature separating x-chains from y-chains, or None."""
    for kind in ("first", "last", "any"):
        S = discriminating_set(xs, ys, kind)
        if S is not None:
            return kind, S
    return None


def derive_decision_list(labelled: list[tuple[Chain, str]], min_support: int = 1):
    """An ordered list of local if-thens, each PURE on the training chains it covers.

    Sequential covering: take the single test that covers the most chains while agreeing on
    all of them, emit it, remove those chains, repeat. Purity is required, not optimised,
    and whatever cannot be covered purely is RETURNED as residue so the caller can abstain
    on it rather than guess. `min_support` is the guard against memorisation and is not
    optional -- see this module's docstring for what happens without it."""
    pool = list(labelled)
    rules: list[tuple[str, str, str]] = []
    while pool:
        best = None
        for kind in ("first", "last", "any"):
            for sym in {v for c, _ in pool for v in KINDS[kind](c)}:
                covered = [(c, l) for c, l in pool if sym in KINDS[kind](c)]
                if len(covered) >= min_support and len({l for _, l in covered}) == 1:
                    if best is None or len(covered) > len(best[3]):
                        best = (kind, sym, covered[0][1], covered)
        if best is None:
            break
        kind, sym, label, covered = best
        rules.append((kind, sym, label))
        pool = [p for p in pool if p not in covered]
    return rules, pool


class PathRules:
    """Rules attached to collision classes: which member a chain's path selects."""

    def __init__(self):
        #: frozenset of a collision class -> ordered list of (kind, symbol, label)
        self.lists: dict[frozenset[str], list[tuple[str, str, str]]] = {}
        #: collision class -> (trigger set, values meaning `label`, label, else-label)
        self.latches: dict[frozenset[str], tuple] = {}
        #: collision class -> (trigger, values, label); fires -> label, else abstain
        self.onesided: dict[frozenset[str], tuple] = {}
        #: chains no pure rule covers; the model abstains on these
        self.residue: dict[frozenset[str], int] = {}
        #: chains carrying more than one gold answer -- the floor, not fitted around
        self.irreducible: set[Chain] = set()
        self._gen: dict = {}

    @classmethod
    def derive(cls, pairs, groups, mode: str = "single",
               min_support: int = 1, gen: dict | None = None) -> "PathRules":
        """One rule per collision class, from labelled chains. Nothing hand-written.

        `mode="single"` (default) emits ONE set-valued test per collision and abstains when
        no single test separates the classes exactly. `mode="list"` emits a decision list of
        singleton tests; it covers more classes and MEMORISES."""
        self = cls()
        self._gen = gen or {}
        by = chains_by_target(pairs)
        for group in groups:
            key = frozenset(group)
            sets = {m: set(by.get(m, [])) for m in group}
            both = {c for m in group for c in sets[m]
                    if sum(1 for n in group if c in sets[n]) > 1}
            self.irreducible |= both
            labelled = sorted((c, m) for m in group for c in sets[m] - both)
            if not labelled:
                continue
            if mode == "list":
                rules, residue = derive_decision_list(labelled, min_support)
                if rules:
                    self.lists[key] = rules
                    self.residue[key] = len(residue)
                continue
            if len(group) != 2:
                continue
            x, y = sorted(group)
            xs = sorted(sets[x] - both)
            ys = sorted(sets[y] - both)
            if min(len(xs), len(ys)) < min_support:
                continue
            # A latch is PREFERRED over a static test when both separate the training data.
            # Training cannot distinguish them -- no training chain crosses a marital
            # boundary and then descends -- so this is a stated inductive bias, not a
            # measurement. The latch is chosen because it reads a DERIVED coordinate rather
            # than a raw position, and so commits to less.
            if gen is not None:
                vocab = {r for c in xs + ys for r in c}
                got = derive_latch(xs, ys, gen, vocab, min_support)
                if got is not None:
                    self.latches[key] = (got[0], got[1], y, x)
                    continue
                got = derive_latch(ys, xs, gen, vocab, min_support)
                if got is not None:
                    self.latches[key] = (got[0], got[1], x, y)
                    continue
            got = derive_rule(xs, ys)
            if got is not None:
                kind, S = got
                self.lists[key] = [(kind, sym, y) for sym in sorted(S)] + [("else", "", x)]
                self.residue[key] = 0
                continue
            got = derive_rule(ys, xs)
            if got is not None:
                kind, S = got
                self.lists[key] = [(kind, sym, x) for sym in sorted(S)] + [("else", "", y)]
                self.residue[key] = 0
                continue
            # Nothing partitions the class. Fall back to claiming only the unanimous region.
            if gen is not None:
                vocab = {r for c in xs + ys for r in c}
                a = derive_onesided_latch(xs, ys, gen, vocab, min_support)
                b = derive_onesided_latch(ys, xs, gen, vocab, min_support)
                pick = max([z for z in ((a, y), (b, x)) if z[0] is not None],
                           key=lambda z: z[0][2], default=None)
                if pick is not None:
                    (T, values, _), label = pick
                    self.onesided[key] = (T, values, label)
        return self

    def decide(self, chain: Chain, group: frozenset[str]) -> str | None:
        """First matching rule wins. None means abstain -- no rule covers this path."""
        if group in self.latches:
            trigger, values, label, other = self.latches[group]
            return label if latch_value(chain, trigger, self._gen) in values else other
        if group in self.onesided:
            trigger, values, label = self.onesided[group]
            return label if latch_value(chain, trigger, self._gen) in values else None
        for kind, sym, label in self.lists.get(group, []):
            if kind == "else" or sym in KINDS[kind](chain):
                return label
        return None

    def select(self, chain: Chain, candidates: list[str]) -> list[str]:
        """Narrow a candidate set using the path. Never adds; only removes."""
        key = frozenset(candidates)
        if key not in self.lists and key not in self.latches and key not in self.onesided:
            return list(candidates)
        pick = self.decide(chain, key)
        return [pick] if pick is not None else list(candidates)

    def explain(self, chain: Chain, group) -> str | None:
        key = frozenset(group)
        if key in self.onesided:
            trigger, values, label = self.onesided[key]
            v = latch_value(chain, trigger, self._gen)
            return (f"after {sorted(trigger)} the next move is '{v}' -> {label}" if v in values
                    else f"after {sorted(trigger)} the next move is '{v}' -> abstain (hold both)")
        if key in self.latches:
            trigger, values, label, other = self.latches[key]
            v = latch_value(chain, trigger, self._gen)
            return (f"after {sorted(trigger)} the next displacement move is '{v}' -> "
                    f"{label if v in values else other}")
        for kind, sym, label in self.lists.get(key, []):
            if kind == "else":
                return f"no test matched -> {label}"
            if sym in KINDS[kind](chain):
                return f"{kind}(chain) is '{sym}' -> {label}"
        return "no rule fires -> abstain" if key in self.lists else None

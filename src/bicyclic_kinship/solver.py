"""The assembly: derive the laws, admit soundly, prefer separately.

KNOWING AND PREFERRING ARE DIFFERENT, and keeping them apart is the whole design.

    solve(chain)   what the derived laws ADMIT. Sound by audit, set-valued whenever the
                   manifold genuinely places two names at one address. Never narrowed by
                   preference.
    best(chain)    the single name the evidence FAVOURS. A preference: it can be wrong, and
                   it can never leave the admitted set.

This is not bookkeeping. Merging the two produced the only genuine errors this system had.
`daughter o grandmother` is an observed training product answered `mother`, and a model
that reads a single witness as a function steps to `{mother}` -- but that address admits
`{mother, mother-in-law}`, because a child's grandmother may be reached through either
parent, and longer training chains land there with `mother-in-law`. Held out, two chains
needed the other name and the system asserted the wrong one. Splitting the two took
soundness to 100% at no cost in single-answer accuracy; what changed is that the system no
longer ASSERTS what it does not know.

The preference stack is deliberately short. Each source may only narrow a set it is given,
so however badly it is ordered the answer stays inside what the laws allow:

    1. `PathRules` -- derived local if-thens over the chain.
    2. training frequency at the chain's address.

Three further sources -- memory of witnessed transitions, term closure over composites,
unit propagation of the observed table -- were measured on all five CLUTRR instances and
contribute EXACTLY NOTHING once the cone lane is present, so they are not here. That is a finding about the model, not a configuration detail: the global law is
tight enough that memory has stopped paying for itself.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .derive import (Chain, Pair, ambiguous_chains, binary_table, table_conflicts,
                     triples_from)
from .pathrules import PathRules
from .system import AxisSystem, collisions

__all__ = ["Solver", "ambiguous_chains", "binary_table", "table_conflicts"]


class Solver:
    """The whole system, trained from `(chain, answer)` pairs over opaque symbols.

    Nothing here knows what a symbol means; `train` is given pairs and derives the rest.

        >>> pairs = [(("father", "father"), "grandfather"), ...]
        >>> s = Solver.train(pairs)
        >>> s.solve(("father", "father", "daughter"))
        ['aunt']
    """

    def __init__(self, system: AxisSystem, rules: PathRules, support: dict | None = None,
                 route=None):
        self.system = system
        self.rules = rules
        #: The opt-in route lane (`route.py`), or None. A law like the others: it narrows.
        self.route = route
        #: address -> Counter of names training witnessed there. A preference, not a law.
        self.support: dict[tuple, Counter] = support or {}

    # ------------------------------------------------------------------ derivation

    @classmethod
    def train(cls, pairs: list[Pair], families=AxisSystem.FAMILIES,
              use_rules: bool = True, validate: bool = True,
              tolerance: float = 0.0, route: str | None = None,
              route_bound: int = 4) -> "Solver":
        """Derive the axes from the atomic products, then the path rules.

        Only length-2 chains enter the derivation. Longer training chains are used twice
        and never memorised: to VALIDATE a derived law, and to count how often each name
        was witnessed at each address.

        `tolerance` above zero says the LABELS are noisy: products are settled by majority
        vote instead of dropped on disagreement, and a law survives while the share of
        training items it contradicts stays under the bound. It trades the audited
        soundness guarantee for staying useful on a corpus that contains mistakes --
        measured in `worlds.py`. Leave it at zero for data you trust.

        `route` switches on the experimental route lane ("move" or "move+label", see
        `route.py`). It is off by default and nothing reported elsewhere uses it."""
        pairs = [(tuple(c), a) for c, a in pairs if c]
        triples = ([(a, b, c) for (a, b), c in binary_table(pairs, vote=True).items()]
                   if tolerance else triples_from(pairs))
        system = AxisSystem.discover(triples, families)
        # A law is derived from binary products but must hold on every training chain.
        # Without this, a family fitted to the single-valued products of a many-valued
        # algebra is simply false, and `solve` returns the empty set.
        if validate:
            system = system.validate(pairs, tolerance)
        gen = system.additive[0] if system.additive else None

        addresses = {s: system.coordinates(s) for s in system.symbols}
        rules = (PathRules.derive(pairs, collisions(addresses), gen=gen)
                 if use_rules else PathRules())

        # How often training witnessed each name AT EACH ADDRESS -- over chains of every
        # length, not only pairs. This is what exposes a one-to-many product: the length-2
        # chain `daughter o grandmother` says `mother` and nothing else, while longer chains
        # landing on the same address say `mother-in-law`.
        support: dict[tuple, Counter] = defaultdict(Counter)
        known = set(system.symbols)
        for chain, answer in pairs:
            if answer in known and all(r in known for r in chain):
                support[system.address(list(chain))][answer] += 1
        law = None
        if route is not None:
            from .route import derive_route
            law = derive_route(pairs, system, read=route, bound=route_bound)
        return cls(system, rules, support, law)

    def ambiguous_addresses(self) -> list[tuple]:
        """Addresses the derived laws place more than one name at. These are exactly the
        products that are relations rather than functions, and no witness makes them
        otherwise."""
        return [a for a in self.support if len(self.system.names_at(a)) > 1]

    # ------------------------------------------------------------------ answering

    def known(self, chain: Chain) -> bool:
        return bool(chain) and all(r in set(self.system.symbols) for r in chain)

    def solve(self, chain) -> list[str]:
        """What the derived laws ADMIT. Sound; never narrowed by preference.

        A set of size one is knowledge. A larger set is the honest statement that the
        manifold places several names at this address and nothing derived separates them.
        The empty set means the laws contradict each other on this chain -- which is what a
        false input fact produces, instead of a confident wrong answer."""
        chain = tuple(chain)
        if not self.known(chain):
            return sorted(self.system.symbols)      # abstain rather than invent
        names = self.system.compose(list(chain))
        if self.route is not None:
            names = [n for n in names if self.route.admits(chain, n)]
        return names

    def best(self, chain) -> str | None:
        """The single name the evidence favours. A PREFERENCE -- it can be wrong."""
        chain = tuple(chain)
        names = self.solve(chain)
        if len(names) <= 1:
            return names[0] if names else None
        narrowed = [n for n in self.rules.select(chain, names) if n in names]
        if len(narrowed) == 1:
            return narrowed[0]
        if narrowed:
            names = narrowed
        counts = self.support.get(self.system.address(list(chain)), {})
        return max(names, key=lambda n: (counts.get(n, 0), n))

    def explain(self, chain) -> list[str]:
        """The derivation, layer by layer -- what each one contributed and why."""
        chain = tuple(chain)
        if not self.known(chain):
            unknown = sorted(set(chain) - set(self.system.symbols))
            return [f"unknown symbols {unknown} -- abstain"]
        names = self.solve(chain)
        out = [f"the laws admit {names}"]
        if len(names) <= 1:
            return out + ["knowledge, not preference"]
        got = self.rules.explain(chain, names)
        if got:
            out.append(f"path rule: {got}")
        counts = self.support.get(self.system.address(list(chain)), {})
        return out + [f"preference picks {self.best(chain)!r} "
                      f"(training support {dict(counts)})"]

    # ------------------------------------------------------------------ measurement

    def score(self, pairs: list[Pair], ambiguous: set | None = None) -> dict:
        """Held-out measurement. `ambiguous` names chains the corpus itself answers two
        ways; those are a floor, not an error, and are counted separately rather than
        excused."""
        ambiguous = ambiguous or set()
        known = set(self.system.symbols)
        rows = [(tuple(c), a) for c, a in pairs
                if c and a in known and all(r in known for r in c)]
        n = len(rows) or 1
        sound = unique = size = top1 = errors = amb = 0
        for chain, gold in rows:
            got = self.solve(chain)
            sound += gold in got
            unique += got == [gold]
            size += len(got)
            picked = self.best(chain)
            top1 += picked == gold
            if picked != gold:
                amb += chain in ambiguous
                errors += chain not in ambiguous
        return {"n": len(rows), "sound": sound / n, "unique": unique / n,
                "top1": top1 / n, "mean_set": size / n, "errors": errors,
                "ambiguous_misses": amb}

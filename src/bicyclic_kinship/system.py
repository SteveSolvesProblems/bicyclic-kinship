"""The manifold implied by a set of composition triples.

Discovered, not declared: `discover()` is the only constructor and it is given nothing but
triples over opaque symbols. Four law families, each a canonical construction with no
freedom to overfit:

    additive    an exact rational null space  (`derive.additive_axes`)
    cone        the bicyclic stack fold       (`cone.cone_axes`)
    right       a union-find partition the LAST step decides  (`projection.projection_axis`)
    left        its mirror, which the FIRST step decides       (`projection.projection_axis`, side="left")

Composition is a FOLD, not a lookup. Each lane folds the chain by its own law and lands
somewhere; `names_at` then reads off every symbol sitting at that address. A chain of
length 100 costs the same per step as a chain of length 2, and nothing longer than an
atomic product was ever learned.

SOUNDNESS. Each law holds on every triple it was derived from, and `validate` additionally
discards any law that a known answer contradicts, at any chain length in the training data.
The lanes are intersected, so the admitted set contains the true answer whenever every
surviving law is true of the algebra. That is an AUDITED property, not a theorem: a family
that survives every training chain can still fail on a chain unlike any of them. What is
structural is that removing a family only LOOSENS the admitted set -- so no subset of the
derivation can be less sound than the whole.
"""
from __future__ import annotations

from fractions import Fraction

from .cone import Cone, cone_axes, cone_fold
from .derive import Triple, additive_axes, symbols_of
from .projection import n_classes, projection_axis


class AxisSystem:
    """The derived laws, and the composition they define."""

    #: Every law family this package derives. Naming them makes the PRIOR explicit: the
    #: coordinates are derived, but this list is a choice, and ablations turn it on and off.
    FAMILIES = ("additive", "cone", "right", "left")

    def __init__(self, additive, right, left, symbols, cone=()):
        self.additive = list(additive)    # list of dict[sym, Fraction]
        self.cone = list(cone)            # list of dict[sym, (u, d)]
        self.right = right                # dict[sym, class] or None
        self.left = left                  # dict[sym, class] or None
        self.symbols = list(symbols)

    @classmethod
    def discover(cls, triples: list[Triple], families=FAMILIES) -> "AxisSystem":
        syms = symbols_of(triples)
        # The additive space is the expensive derivation and the cone search needs it too,
        # since a cone's difference `d - u` must be one of its members. Derive it once.
        spine = additive_axes(triples) if {"additive", "cone"} & set(families) else []
        add = spine if "additive" in families else []
        cone = ([c for c in cone_axes(triples, additive=spine) if all(s in c for s in syms)]
                if "cone" in families else [])
        right = projection_axis(triples, "right") if "right" in families else None
        left = projection_axis(triples, "left") if "left" in families else None
        return cls(add,
                   right if right is not None and n_classes(right) > 1 else None,
                   left if left is not None and n_classes(left) > 1 else None,
                   syms, cone)

    # ---------------------------------------------------------------- coordinates

    def coordinates(self, sym: str) -> tuple:
        """A symbol's address: one coordinate per discovered axis."""
        out: list = [g[sym] for g in self.additive]
        out.extend(c[sym] for c in self.cone)
        if self.right is not None:
            out.append(self.right[sym])
        if self.left is not None:
            out.append(self.left[sym])
        return tuple(out)

    def address(self, chain: list[str]) -> tuple:
        """Where a chain lands. Every lane here folds to a POINT, so there is one notion of
        position and it is what both the answer and the preference evidence are keyed on."""
        add = tuple(sum((g[r] for r in chain), Fraction(0)) for g in self.additive)
        # EVERY cone solution must agree: the data pins the assignment only up to a family.
        cone = tuple(cone_fold([c[r] for r in chain]) for c in self.cone)
        r = self.right[chain[-1]] if self.right is not None else None
        l = self.left[chain[0]] if self.left is not None else None
        return (add, cone, r, l)

    def names_at(self, address: tuple) -> list[str]:
        """Every symbol sitting at an address.

        Empty is normal and is not an error: the manifold is larger than the vocabulary, so
        a reachable position need not have a name. Empty from a chain built out of TRUE
        facts means the laws contradict each other there -- which is how a false input fact
        is detected rather than answered."""
        add, cone, r, l = address
        out = []
        for s in self.symbols:
            if any(g[s] != w for g, w in zip(self.additive, add)):
                continue
            if any(c[s] != w for c, w in zip(self.cone, cone)):
                continue
            if r is not None and self.right[s] != r:
                continue
            if l is not None and self.left[s] != l:
                continue
            out.append(s)
        return out

    def compose(self, chain: list[str]) -> list[str]:
        """Every symbol still consistent with composing the chain, using ONLY exact laws."""
        return self.names_at(self.address(chain))

    # ---------------------------------------------------------------- audit

    def validate(self, pairs, tolerance: float = 0.0) -> "AxisSystem":
        """Drop every derived axis that a KNOWN answer contradicts.

        Derivation only ever sees binary products, so soundness off-sample is contingent on
        the derived laws happening to hold. Each family is an independent constraint and
        `compose` intersects them, so a family is valid exactly when it admits the known
        answer on every training item -- of any length, not just the products it was derived
        from. A law that contradicts the data it was learned from is not a law.

        This does not make soundness a theorem; it makes it an audited property.

        `tolerance` is for corpora whose LABELS are noisy. At zero -- the default, and the
        only setting that keeps the guarantee this class advertises -- a single
        contradicting item retires a family. That is correct when the data is trustworthy
        and catastrophic when it is not: one mislabelled answer in a thousand is enough to
        retire every law and leave the system abstaining on everything. Above zero, a
        family survives while the fraction of items it contradicts stays under the bound,
        and soundness stops being audited against all of the data. Set it only when the
        input is known to be noisy, and read the admitted sets as likely rather than
        guaranteed."""
        keep_add, keep_cone = [], []
        known = set(self.symbols)
        items = [(list(c), a) for c, a in pairs if c and a in known
                 and all(r in known for r in c)]
        n = len(items) or 1

        def holds(test) -> bool:
            """True while the share of items the law contradicts stays within tolerance."""
            bad = 0
            for c, a in items:
                if not test(c, a):
                    bad += 1
                    if bad > tolerance * n:
                        return False
            return True

        for g in self.additive:
            if holds(lambda c, a, g=g: sum((g[r] for r in c), Fraction(0)) == g[a]):
                keep_add.append(g)
        for cone in self.cone:
            if holds(lambda c, a, cone=cone: cone_fold([cone[r] for r in c]) == cone[a]):
                keep_cone.append(cone)
        ok_r = self.right is not None and holds(
            lambda c, a: self.right[c[-1]] == self.right[a])
        ok_l = self.left is not None and holds(
            lambda c, a: self.left[c[0]] == self.left[a])
        return AxisSystem(keep_add, self.right if ok_r else None,
                          self.left if ok_l else None, self.symbols, keep_cone)

    def describe(self) -> str:
        parts = [f"{len(self.additive)} additive"]
        if self.cone:
            parts.append(f"{len(self.cone)} cone")
        if self.right is not None:
            parts.append(f"right-projection with {n_classes(self.right)} classes")
        if self.left is not None:
            parts.append(f"left-projection with {n_classes(self.left)} classes")
        return f"AxisSystem over {len(self.symbols)} symbols: " + ", ".join(parts)


def collisions(addresses: dict[str, tuple]) -> list[list[str]]:
    """Symbols the manifold cannot tell apart. These are exactly the ambiguous answers."""
    groups: dict[tuple, list[str]] = {}
    for s, a in addresses.items():
        groups.setdefault(a, []).append(s)
    return [sorted(g) for g in groups.values() if len(g) > 1]

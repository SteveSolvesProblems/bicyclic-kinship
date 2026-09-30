"""The bicyclic cone: a stack coordinate `(u, d)` with cancellation.

A symbol carries a pair `(u, d)` -- how far a path RISES and how far it FALLS -- and
composition translates and then folds back onto the boundary of the positive quadrant:

    (u1, d1) . (u2, d2) = ( u1 + max(0, u2 - d1),  d2 + max(0, d1 - u2) )

This is the bicyclic monoid: push and pop on a stack, or a canonicalised path in a tree.
It is associative, it is a FORMULA on coordinates rather than a table lookup, and it is
exactly the structure a group cannot carry.

WHY IT IS NEEDED. Rotations and constant maps cannot express CANCELLATION -- that a path
up and then back down partly annihilates, so that a mother's son is a brother rather than
a "mother-son". Cancellation is not a group operation and not a constant map, so no amount
of searching those families finds it. Without this lane, everything it should explain is
absorbed by the projection axis, which is a table walk wearing a coordinate's clothes and
is measurably unsound off-sample.

Its linear shadow `d - u` IS an additive axis -- which is the proof that generation was
only ever the visible part of the cone, and the reason the search below derives `u` given
the additive axis instead of searching a plane.

DERIVED, NOT FITTED. `cone_axes` returns every assignment within a bound that reproduces
EVERY observed product exactly under the fold. An assignment that misses one product is
discarded, not scored. Assigning two operands forces their product, so the search is
mostly unit propagation and backtracking is shallow.
"""
from __future__ import annotations

from math import gcd as math_gcd

from .derive import Triple, additive_axes, symbols_of

Cone = tuple[int, int]

#: Ceiling on the escalation in `cone_axes`, and it has to be MEASURED rather than picked
#: generously. A world with no cone law pays every depth, and the cost of one depth grows
#: with the vocabulary: the 441-symbol lattice has additive peaks of 20 and 170, so it
#: yields no difference candidate at all until depth 20 and then costs ~11s per depth --
#: raising this to 24 put 76s into a derivation that is otherwise 2.5s. Every world here
#: needs a depth equal to its own reach (paths capped at k need exactly k, CLUTRR needs 2,
#: the blood tree 4), so 12 clears them all with margin and stops short of that cliff. A
#: world needing more simply gets no cone lane, which is sound -- fewer laws admit more.
CONE_DEPTH_CAP = 12

#: Depths to take BEYOND the shallowest one that yields a law. A law that fits at depth d
#: still fits at d+1, so a deeper search returns a SUPERSET, and since `names_at` admits a
#: symbol only where every cone solution agrees, more laws NARROW the answer.
#:
#: Narrowing is not automatically safe, and the margin is one step for that reason. Fitting
#: every OBSERVED product is not the same as being valid on every TRUE one, so an admitted
#: law can be wrong off-sample, and intersecting more of them can exclude the true answer
#: rather than a false one. That is not hypothetical: on a world where composition is
#: multi-valued -- a tree where a sibling's sibling may be yourself -- `binary_table` drops
#: the conflicting products, nothing constrains the laws there, and unanimity across them
#: returns the EMPTY set on 78% of held-out chains. Neither direction is free: fewer laws
#: risk admitting a falsehood, more risk excluding the truth. What catches it is not
#: unanimity and not another lane but the AUDIT -- all 34 such laws contradict some training
#: item, every one is retired, and soundness goes back to 100% with no cone lane at all.
#: At the least depth alone CLUTRR pins ONE assignment, and that one assignment asserts
#: `husband o wife = wife`, which is false: it is SELF, and CLUTRR has no word for it.
#: Neither spouse product is ever observed, so nothing retires the law during derivation.
#: One depth deeper the data admits four, they disagree about exactly that cancellation,
#: and the system goes back to answering "none of these". Held-out scores are identical
#: either way; the margin is bought entirely off-benchmark.
CONE_DEPTH_MARGIN = 1


def cone_compose(a: Cone, b: Cone) -> Cone:
    """`(u1,d1) . (u2,d2)`. Translate, then fold onto the boundary of the quadrant."""
    (u1, d1), (u2, d2) = a, b
    return (u1 + max(0, u2 - d1), d2 + max(0, d1 - u2))


def cone_fold(chain: list[Cone]) -> Cone:
    """The law is associative, so a chain reduces left to right with no ambiguity.

    This is the whole of length extrapolation: a chain of any length folds by the same
    two-argument formula, so a model derived from products of length two composes a chain
    of length 100 with no table, no decay and no retraining."""
    out = chain[0]
    for nxt in chain[1:]:
        out = cone_compose(out, nxt)
    return out


def _difference_candidates(additive, symbols, bound):
    """Integer maps `d - u` can take. It must be additive, which is what makes this finite.

    `d(a o b) - u(a o b) = (d_a - u_a) + (d_b - u_b)` follows from the fold by cases, so the
    difference is a member of the additive space and nothing outside it need be tried."""
    if not additive:
        return []
    prims = []
    for g in additive:
        den = 1
        for v in g.values():
            den = den * v.denominator // math_gcd(den, v.denominator)
        ints = {s: int(v * den) for s, v in g.items()}
        common = 0
        for v in ints.values():
            common = math_gcd(common, abs(v))
        prims.append({s: v // common for s, v in ints.items()} if common else ints)
    out, seen = [], set()
    for p in prims:
        peak = max((abs(v) for v in p.values()), default=0) or 1
        for m in range(1, bound // peak + 1):
            for sign in (1, -1):
                cand = {s: sign * m * p.get(s, 0) for s in symbols}
                key = tuple(sorted(cand.items()))
                if key not in seen:
                    seen.add(key)
                    out.append(cand)
    return out


def _cone_gauge(assign):
    """A global shift of every `(u,d)` commutes with the fold, so it is pure gauge.
    Fixed by pushing the assignment down until some symbol touches the boundary."""
    k = min(min(u, d) for u, d in assign.values())
    return {s: (u - k, d - k) for s, (u, d) in assign.items()}


def cone_axes(triples: list[Triple], bound: int | None = None, limit: int = 200,
              additive=None) -> list[dict[str, Cone]]:
    """Every `(u,d)` assignment reproducing all triples under the fold, up to gauge.

    `bound` DOES TWO UNRELATED JOBS, which is why it must not be a fixed constant. It is
    the depth a coordinate may reach (`u` ranges over it, and `propagate` rejects past it),
    which has to SCALE with the vocabulary -- a world whose fragments pop five cannot be
    expressed below five, and at four the lane does not narrow, it VANISHES. It is also the
    multiple of the additive axis the difference `d - u` may take, which must stay SMALL --
    every extra multiple is another whole family of solutions, and since every returned
    solution must agree before a composite is claimed, more solutions is a strictly weaker
    lane. One knob, two jobs pulling opposite ways.

    So `bound=None` (the default) reads the depth off the data: escalate from 1 and return
    the first depth that yields anything. That is the smallest solution set the data admits
    and the shallowest law that explains it, and it is better on both counts rather than a
    trade. On CLUTRR it returns ONE assignment where a fixed bound of 4 returned ten.

    Returns [] when the algebra admits no such law -- including whenever it admits no
    additive axis, since the difference `d - u` would have to be one. Pass `additive` when
    the caller has already derived that space; it is the expensive part and deriving it
    twice is the single largest avoidable cost in the whole system.

    Several solutions are still normal on some worlds and are NOT a tie to break: the data
    may pin the assignment only up to a family, and every member must agree before a
    composite is claimed (`AxisSystem.address`). Claiming a product one member denies would
    not be sound."""
    if bound is None:
        if additive is None:
            additive = additive_axes(triples)
        for b in range(1, CONE_DEPTH_CAP + 1):
            if cone_axes(triples, bound=b, limit=limit, additive=additive):
                deep = min(b + CONE_DEPTH_MARGIN, CONE_DEPTH_CAP)
                return cone_axes(triples, bound=deep, limit=limit, additive=additive)
        return []
    syms = symbols_of(triples)
    sols: list[dict[str, Cone]] = []
    if additive is None:
        additive = additive_axes(triples)
    for diff in _difference_candidates(additive, syms, bound):
        if any(s not in diff for s in syms):
            continue
        order = sorted(syms, key=lambda s: (-sum(s in (a, b) for a, b, _ in triples), s))

        def propagate(assign):
            assign = dict(assign)
            changed = True
            while changed:
                changed = False
                for a, b, c in triples:
                    if a in assign and b in assign:
                        v = cone_compose(assign[a], assign[b])
                        if c in assign:
                            if assign[c] != v:
                                return None
                        elif max(v) > bound:
                            return None
                        else:
                            assign[c], changed = v, True
            return assign

        def go(i, assign):
            if len(sols) >= limit:
                return
            while i < len(order) and order[i] in assign:
                i += 1
            if i == len(order):
                got = _cone_gauge(assign)
                if got == assign and got not in sols:      # already gauge-fixed
                    sols.append(got)
                return
            s = order[i]
            for u in range(bound + 1):
                if u + diff[s] < 0 or u + diff[s] > bound:
                    continue
                nxt = propagate({**assign, s: (u, u + diff[s])})
                if nxt is not None:
                    go(i + 1, nxt)

        go(0, {})
    return sols

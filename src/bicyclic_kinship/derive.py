"""Exact derivation of additive coordinates from composition triples.

Everything here takes `(a, b, c)` triples meaning `a o b = c` over OPAQUE symbols. No
function in this package knows what a symbol means; kinship is an input, not a subject.

An AXIS is a map from symbols to coordinates together with a law saying how the coordinate
of a composite is determined:

    additive     g(a o b) = g(a) + g(b)      an invertible, order-free displacement
    cone         (u, d) composed with CANCELLATION -- see `cone.py`
    projection   g(a o b) = g(b) (or g(a))   a terminal class -- see `projection.py`

The additive derivation is an exact rational null space: every `g` satisfying the law on
all triples at once, as a basis. Dimension 0 means the algebra admits no additive
invariant. Dimension 1 means the coordinate is FORCED up to scale rather than chosen --
there is nothing to fit and nothing to tune. Arithmetic is `Fraction` throughout, so an
invariant is exact rather than a least-squares fit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction

Triple = tuple[str, str, str]           # (a, b, c) asserting a o b = c
_ZERO = Fraction(0)
Chain = tuple[str, ...]
Pair = tuple[Chain, str]


# ------------------------------------------------------------------ exact linear algebra

def _reduce_sparse(rows: list[dict[int, Fraction]]) -> dict[int, dict[int, Fraction]]:
    """Reduced row echelon form, kept sparse and built one row at a time.

    Every row here comes from a triple `a o b = c`, so it has at most three non-zero
    entries however large the vocabulary is, and there are far more rows than the rank can
    ever be. Eliminating a dense matrix therefore spends nearly all of its time on zeros:
    on a 225-symbol world it is 28561 rows of 225 columns and takes a minute, while the
    answer is a two-dimensional null space.

    So rows are reduced against the basis as they arrive and thrown away when they collapse
    to nothing, which is what most of them do. The result is the same reduced echelon form
    -- exact `Fraction` arithmetic throughout, no tolerance, no pivoting choice to make --
    keyed by pivot column."""
    pivots: dict[int, dict[int, Fraction]] = {}
    for row in rows:
        r = {c: v for c, v in row.items() if v}
        # Eliminate EVERY pivot column the row touches, not merely its leading one: a row
        # whose leading column is free can still carry a pivot column further right, and
        # leaving it there is not echelon form.
        while True:
            c = next((x for x in sorted(r) if x in pivots), None)
            if c is None:
                break
            f, pr = r[c], pivots[c]
            for cc, v in pr.items():
                nv = r.get(cc, _ZERO) - f * v
                if nv:
                    r[cc] = nv
                else:
                    r.pop(cc, None)
        if not r:
            continue
        c = min(r)
        inv = r[c]
        r = {cc: v / inv for cc, v in r.items()}
        for pr in pivots.values():                 # keep the basis fully reduced
            f = pr.pop(c, None)
            if f is None:
                continue
            for cc, v in r.items():
                if cc == c:
                    continue
                nv = pr.get(cc, _ZERO) - f * v
                if nv:
                    pr[cc] = nv
                else:
                    pr.pop(cc, None)
        pivots[c] = r
    return pivots


def _null_space_of(rows: list[dict[int, Fraction]], ncols: int) -> list[list[Fraction]]:
    """Basis of {x : Mx = 0} for sparse rows, exactly. One vector per free column."""
    pivots = _reduce_sparse(rows)
    basis = []
    for f in (c for c in range(ncols) if c not in pivots):
        v = [Fraction(0)] * ncols
        v[f] = Fraction(1)
        for pc, pr in pivots.items():
            v[pc] = -pr.get(f, _ZERO)
        basis.append(v)
    return basis


def _rref(rows: list[list[Fraction]], ncols: int):
    """Reduced row echelon form over the rationals. Exact: no floating point.

    Kept as the reference implementation the sparse one is checked against."""
    mat = [r[:] for r in rows]
    pivots: list[int] = []
    r = 0
    for c in range(ncols):
        piv = next((i for i in range(r, len(mat)) if mat[i][c] != 0), None)
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        inv = mat[r][c]
        mat[r] = [x / inv for x in mat[r]]
        for i in range(len(mat)):
            if i != r and mat[i][c] != 0:
                f = mat[i][c]
                mat[i] = [x - f * y for x, y in zip(mat[i], mat[r])]
        pivots.append(c)
        r += 1
        if r == len(mat):
            break
    return mat[:r], pivots


def _null_space(rows, ncols) -> list[list[Fraction]]:
    """Basis of {x : Mx = 0}, exactly."""
    red, pivots = _rref(rows, ncols)
    free = [c for c in range(ncols) if c not in pivots]
    basis = []
    for f in free:
        v = [Fraction(0)] * ncols
        v[f] = Fraction(1)
        for i, p in enumerate(pivots):
            v[p] = -red[i][f]
        basis.append(v)
    return basis


# ---------------------------------------------------------------------------- the axis

def symbols_of(triples: list[Triple]) -> list[str]:
    """Every symbol appearing anywhere in the triples, sorted."""
    return sorted({s for t in triples for s in t})


def additive_axes(triples: list[Triple]) -> list[dict[str, Fraction]]:
    """Every `g` with `g(a) + g(b) = g(c)` on all triples. A basis for the space."""
    syms = symbols_of(triples)
    idx = {s: i for i, s in enumerate(syms)}
    rows = []
    for a, b, c in triples:
        row: dict[int, Fraction] = {}
        for k, sign in ((idx[a], 1), (idx[b], 1), (idx[c], -1)):
            row[k] = row.get(k, _ZERO) + sign
        rows.append({k: v for k, v in row.items() if v})
    return [dict(zip(syms, v)) for v in _null_space_of(rows, len(syms))]


# ------------------------------------------------------------- observations -> triples

def binary_table(pairs: list[Pair], vote: bool = False) -> dict[tuple[str, str], str]:
    """The observed atomic products: length-2 chains only.

    By default a multi-valued entry is a contradiction in the data, not something to
    average, so it is dropped and reported by `table_conflicts` rather than voted on. On
    CLUTRR's training split this yields exactly 62 products over 20 symbols -- the entire
    input to the derivation. Longer training chains are never memorised; they are used
    only to VALIDATE a derived law (`AxisSystem.validate`) and to count preference evidence.

    `vote=True` takes the majority answer instead of dropping. That is the right reading
    when answers are NOISY rather than genuinely many-valued -- with a few dozen witnesses
    per product, one mislabelled witness otherwise deletes the product entirely. It is not
    the default because on clean data a product with two answers is telling you something
    true, and voting silences it."""
    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for chain, answer in pairs:
        if len(chain) == 2:
            votes[(chain[0], chain[1])][answer] += 1
    if vote:
        return {k: v.most_common(1)[0][0] for k, v in votes.items()}
    return {k: next(iter(v)) for k, v in votes.items() if len(v) == 1}


def table_conflicts(pairs: list[Pair]) -> list[tuple[str, str]]:
    """Products the data answers more than one way. Dropped by `binary_table`."""
    votes: dict[tuple[str, str], set] = defaultdict(set)
    for chain, answer in pairs:
        if len(chain) == 2:
            votes[(chain[0], chain[1])].add(answer)
    return sorted(k for k, v in votes.items() if len(v) > 1)


def triples_from(pairs: list[Pair]) -> list[Triple]:
    """`(a, b, c)` triples from observed chains -- the derivation's only input."""
    return [(a, b, c) for (a, b), c in binary_table(pairs).items()]


def ambiguous_chains(*splits) -> set[Chain]:
    """Chains the data itself answers more than one way -- the irreducible floor.

    No deterministic function of the relation chain can answer these both ways, so they
    bound what ANY chain-only model can score. Reported, not excused."""
    gold: dict[Chain, set] = defaultdict(set)
    for split in splits:
        for chain, answer in split:
            gold[tuple(chain)].add(answer)
    return {c for c, a in gold.items() if len(a) > 1}

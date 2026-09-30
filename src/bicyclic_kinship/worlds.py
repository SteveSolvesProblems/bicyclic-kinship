"""Worlds beyond CLUTRR: a real family tree, deeper chains, and other algebras.

CLUTRR is one generator. Everything here is a DIFFERENT source of truth, so that the
solver's answers are checked against something that does not compose chains at all:

    genealogy   an actual family tree. Walks of any length are taken through it, and the
                true relation between the endpoints is computed from GRAPH DISTANCES --
                lowest common ancestor, ascent, descent -- never by folding the chain.
                The solver sees only the sequence of relation words.
    paths       filesystem path normalisation. The truth is `posixpath.normpath`.
    grid        displacement on a 2-D lattice. The truth is vector addition, and the
                additive lane has to find dimension TWO, which CLUTRR never exercises.

The generator in each world knows what its symbols mean. The solver never does: it is
given products over opaque strings and must derive the rest.
"""
from __future__ import annotations

import posixpath
import random
from collections import defaultdict

# --------------------------------------------------------------------------- genealogy

#: (ascent, descent) -> the pair of names, female first. The GENERATOR's knowledge of
#: kinship, used only to label the ground truth -- the solver is never shown this.
BLOOD = {
    (1, 0): ("mother", "father"),
    (0, 1): ("daughter", "son"),
    (1, 1): ("sister", "brother"),
    (2, 0): ("grandmother", "grandfather"),
    (0, 2): ("granddaughter", "grandson"),
    (2, 1): ("aunt", "uncle"),
    (1, 2): ("niece", "nephew"),
}


class Family:
    """A family tree, built generation by generation. Nobody marries a blood relative, so
    every pair of people stands in at most one relation and the ground truth is a
    function rather than a choice."""

    def __init__(self, generations: int = 5, children: tuple[int, int] = (2, 3),
                 seed: int = 0):
        self.rng = random.Random(seed)
        self.female: list[bool] = []
        self.father: list[int | None] = []
        self.mother: list[int | None] = []
        self.spouse: list[int | None] = []
        self.kids: list[list[int]] = []
        couples = [self._couple()]
        for _ in range(generations - 1):
            nxt = []
            for wife, husband in couples:
                for _ in range(self.rng.randint(*children)):
                    child = self._person(self.rng.random() < .5, husband, wife)
                    self.kids[husband].append(child)
                    self.kids[wife].append(child)
                    partner = self._person(not self.female[child], None, None)
                    self._marry(child, partner)
                    nxt.append((child, partner) if self.female[child]
                               else (partner, child))
            couples = nxt

    def _person(self, female: bool, father, mother) -> int:
        self.female.append(female)
        self.father.append(father)
        self.mother.append(mother)
        self.spouse.append(None)
        self.kids.append([])
        return len(self.female) - 1

    def _couple(self):
        w = self._person(True, None, None)
        h = self._person(False, None, None)
        self._marry(w, h)
        return (w, h)

    def _marry(self, a: int, b: int) -> None:
        self.spouse[a], self.spouse[b] = b, a

    def __len__(self) -> int:
        return len(self.female)

    # ------------------------------------------------------------------ structure

    def parents(self, p: int) -> list[int]:
        return [x for x in (self.father[p], self.mother[p]) if x is not None]

    def siblings(self, p: int) -> list[int]:
        f = self.father[p]
        return [] if f is None else [k for k in self.kids[f] if k != p]

    def ancestors(self, p: int) -> dict[int, int]:
        """Every blood ancestor of `p`, with its distance. `p` is its own at distance 0."""
        out, frontier, d = {p: 0}, [p], 0
        while frontier:
            d += 1
            nxt = []
            for x in frontier:
                for a in self.parents(x):
                    if a not in out:
                        out[a] = d
                        nxt.append(a)
            frontier = nxt
        return out

    def blood(self, a: int, b: int) -> tuple[int, int] | None:
        """(ascent, descent) between two people, through their nearest common ancestor.

        Pure graph distance: this never composes a relation word, which is what makes it
        an independent check on a solver that only ever composes relation words."""
        up_a, up_b = self.ancestors(a), self.ancestors(b)
        common = set(up_a) & set(up_b)
        if not common:
            return None
        best = min(common, key=lambda c: (up_a[c] + up_b[c], up_a[c]))
        return up_a[best], up_b[best]

    def term(self, a: int, b: int) -> str | None:
        """What `b` is to `a`, in CLUTRR's twenty words. None when those words cannot say.

        The order of the checks is the definition, not a heuristic: a spouse first, then a
        blood tie, then the two in-law families CLUTRR names."""
        if a == b or b is None:
            return None
        fem = self.female[b]
        if self.spouse[a] == b:
            return "wife" if fem else "husband"
        got = self.blood(a, b)
        if got in BLOOD:
            return BLOOD[got][0 if fem else 1]
        if got is not None:
            return None                                  # blood, but past the vocabulary
        partner = self.spouse[b]
        if partner is not None:
            if self.blood(a, partner) == (0, 1):         # your child's spouse
                return "daughter-in-law" if fem else "son-in-law"
        mine = self.spouse[a]
        if mine is not None and self.blood(mine, b) == (1, 0):   # your spouse's parent
            return "mother-in-law" if fem else "father-in-law"
        return None

    # ------------------------------------------------------------------ walks

    def steps(self, p: int) -> list[tuple[str, int]]:
        """Every single step available from `p`, as (relation word, person)."""
        out: list[tuple[str, int]] = []
        for x in self.parents(p):
            out.append(("mother" if self.female[x] else "father", x))
            for g in self.parents(x):
                out.append(("grandmother" if self.female[g] else "grandfather", g))
            for s in self.siblings(x):
                out.append(("aunt" if self.female[s] else "uncle", s))
        for k in self.kids[p]:
            out.append(("daughter" if self.female[k] else "son", k))
            for g in self.kids[k]:
                out.append(("granddaughter" if self.female[g] else "grandson", g))
            if self.spouse[k] is not None:
                x = self.spouse[k]
                out.append(("daughter-in-law" if self.female[x] else "son-in-law", x))
        for s in self.siblings(p):
            out.append(("sister" if self.female[s] else "brother", s))
            for n in self.kids[s]:
                out.append(("niece" if self.female[n] else "nephew", n))
        if self.spouse[p] is not None:
            x = self.spouse[p]
            out.append(("wife" if self.female[x] else "husband", x))
            for g in self.parents(x):
                out.append(("mother-in-law" if self.female[g] else "father-in-law", g))
        return out

    def atomic_steps(self, p: int) -> list[tuple[str, int]]:
        """The PRIMITIVE moves only: one tree edge each.

        Compound words -- grandmother, uncle, niece -- are two edges in one step, and a
        chain of them can re-converge on a person it has already passed through without
        the chain recording it. `father o grandson` is your SON when that grandson is your
        own child and your NEPHEW otherwise, so the truth stops being a function of the
        chain. CLUTRR's own generator avoids this; restricting to atomic steps is how this
        world avoids it too."""
        out: list[tuple[str, int]] = []
        for x in self.parents(p):
            out.append(("mother" if self.female[x] else "father", x))
        for k in self.kids[p]:
            out.append(("daughter" if self.female[k] else "son", k))
        for sib in self.siblings(p):
            out.append(("sister" if self.female[sib] else "brother", sib))
        if self.spouse[p] is not None:
            x = self.spouse[p]
            out.append(("wife" if self.female[x] else "husband", x))
        return out

    def simple_walk(self, length: int, start: int | None = None, tries: int = 60):
        """A walk of `length` ATOMIC steps that never revisits a person.

        No revisit means no re-convergence, which is exactly the condition under which the
        relation chain determines the answer. Returns (start, chain, end), or None if the
        tree ran out of room."""
        for _ in range(tries):
            p = self.rng.randrange(len(self)) if start is None else start
            origin, seen, chain = p, {p}, []
            while len(chain) < length:
                opts = [(w, q) for w, q in self.atomic_steps(p) if q not in seen]
                if not opts:
                    break
                word, nxt = self.rng.choice(opts)
                chain.append(word)
                seen.add(nxt)
                p = nxt
            if len(chain) == length:
                return origin, tuple(chain), p
        return None

    def walk(self, length: int, start: int | None = None):
        """A random walk of `length` steps. Returns (start, chain, end).

        The chain is what a solver is given. `start` and `end` are what the ground truth
        is computed from, and are never shown to it."""
        p = self.rng.randrange(len(self)) if start is None else start
        origin, chain = p, []
        for _ in range(length):
            opts = self.steps(p)
            if not opts:
                break
            word, nxt = self.rng.choice(opts)
            chain.append(word)
            p = nxt
        return origin, tuple(chain), p


# ------------------------------------------------------- extended kinship vocabulary

def kin_name(up: int, down: int, female: bool, bound: int = 4) -> str | None:
    """The English name for an (ascent, descent) pair, or None past `bound`.

    Anthropology's naming rule, written out: ancestors and descendants get "great-"
    prefixes, one sibling step each way is a sibling, one on either side is an
    uncle/aunt or a nephew/niece, and two or more on both sides is a cousin whose DEGREE
    is one less than the shorter leg and whose REMOVE is the difference. English uses the
    same words for a cousin in either direction; these keep the direction, because a
    symbol that names two different coordinates is not a symbol.

    This is the generator's knowledge. The solver is shown the names and never the rule."""
    if up > bound or down > bound:
        return None
    g = lambda f, m: f if female else m
    greats = lambda n: "great-" * (n - 2) + "grand"
    if up == down == 0:
        return "self"
    if down == 0:
        return g("mother", "father") if up == 1 else greats(up) + g("mother", "father")
    if up == 0:
        return g("daughter", "son") if down == 1 else greats(down) + g("daughter", "son")
    if up == down == 1:
        return g("sister", "brother")
    if down == 1:
        return "great-" * (up - 2) + g("aunt", "uncle")
    if up == 1:
        return "great-" * (down - 2) + g("niece", "nephew")
    degree, removed = min(up, down) - 1, abs(up - down)
    ordinal = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
               6: "sixth", 7: "seventh", 8: "eighth"}[degree]
    name = f"{ordinal} cousin"
    if removed:
        name += f" {removed}x removed {'up' if up > down else 'down'}"
    return name + g(" (f)", " (m)")


class BloodWorld:
    """A family tree walked by blood steps only, named by `kin_name`.

    Two things this buys that CLUTRR cannot. Its vocabulary is open -- cousins, greats and
    removes are generated by a rule rather than listed, so a chain that wanders far still
    has a name and depth can be pushed. And it contains full CANCELLATION: `son o mother`
    is SELF, a product CLUTRR never shows, which is the one observation that was known to
    pin the cone assignment uniquely."""

    #: The step words a walk may take: CLUTRR's own, minus spouses and in-laws.
    STEPS = ("parent", "child", "sibling", "grandparent", "grandchild",
             "uncle/aunt", "nephew/niece")

    #: How the generator names relatives. "exact" keeps the direction of a remove and the
    #: sex of a cousin, so every word names one (ascent, descent, sex). "no-direction"
    #: drops " up"/" down", as English does: "first cousin 1x removed" then names two
    #: coordinates. "english" also drops the sex of a cousin, as English does.
    NAMINGS = ("exact", "no-direction", "english")

    def __init__(self, family: "Family", bound: int = 4, naming: str = "exact"):
        if naming not in self.NAMINGS:
            raise ValueError(f"naming must be one of {self.NAMINGS}")
        self.fam = family
        self.bound = bound
        self.naming = naming

    def name(self, up: int, down: int, female: bool) -> str | None:
        """The generator's word for a relative, under this world's naming."""
        n = kin_name(up, down, female, self.bound)
        if n is None or self.naming == "exact":
            return n
        n = n.replace(" up", "").replace(" down", "")
        if self.naming == "english" and "cousin" in n:
            n = n.replace(" (f)", "").replace(" (m)", "")
        return n

    def steps(self, p: int, visited: frozenset[int] = frozenset()):
        """Every step available from `p`, as (word, target, people passed through).

        These are CLUTRR's own step words -- parent, child, sibling, grandparent,
        grandchild, uncle/aunt, nephew/niece -- and each carries the INTERMEDIATE people
        it walks through. Tracking them is what keeps a chain honest: a step that passes
        back through someone the walk has already met would let the chain re-converge, and
        then the answer stops being a function of the chain."""
        f, out = self.fam, []

        def add(target, via, up, down):
            if target in visited or any(v in visited for v in via) or target == p:
                return
            name = self.name(up, down, f.female[target])
            if name is not None:
                out.append((name, target, frozenset(via) | {target}))

        for x in f.parents(p):
            add(x, [], 1, 0)
            for y in f.parents(x):
                add(y, [x], 2, 0)
            for sib in f.siblings(x):
                add(sib, [x], 2, 1)
        for k in f.kids[p]:
            add(k, [], 0, 1)
            for g in f.kids[k]:
                add(g, [k], 0, 2)
        for sib in f.siblings(p):
            add(sib, [], 1, 1)
            for n in f.kids[sib]:
                add(n, [sib], 1, 2)
        return out

    def term(self, a: int, b: int) -> str | None:
        got = self.fam.blood(a, b)
        return None if got is None else self.name(*got, self.fam.female[b])

    def walk(self, length: int, tries: int = 12):
        """A walk of `length` steps that never meets the same person twice.

        Returns (chain, true term) -- the term computed from graph distance between the
        endpoints, never by composing the chain -- or None if the tree ran out of room."""
        f = self.fam
        for _ in range(tries):
            p = f.rng.randrange(len(f))
            origin, seen, chain = p, {p}, []
            while len(chain) < length:
                opts = self.steps(p, frozenset(seen))
                if not opts:
                    break
                word, nxt, via = f.rng.choice(opts)
                chain.append(word)
                seen |= via
                p = nxt
            if len(chain) == length:
                return tuple(chain), self.term(origin, p)
        return None

    def products(self, samples: int = 40000) -> list[tuple[tuple[str, ...], str]]:
        """Observed products: two steps, and the name of where they land."""
        out = {}
        for _ in range(samples):
            got = self.walk(2)
            if got and got[1] is not None:
                out[got[0]] = got[1]
        return sorted(out.items())


class Lineage:
    """An endless family, grown on demand, for walks of any length.

    `Family` is built in advance, and a finite tree runs out: a walk that never meets the
    same person twice cannot get past about twenty steps. Here nobody exists until a walk
    needs them -- a parent is created the first time someone asks for one, and a fresh child
    can always be born -- so a walk of ten thousand steps never has to revisit anyone.

    Each person records ONE blood parent. The other parent married in from outside and no
    blood step reaches them, so the family is a tree and the truth is exact graph distance
    to the nearest common ancestor. Steps are CLUTRR's blood words, named by `BLOOD`."""

    #: (ascent, descent) of each step a walk may take, and the intermediate it passes.
    STEPS = ((1, 0), (2, 0), (0, 1), (0, 2), (1, 1), (2, 1), (1, 2))

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.parent: list[int | None] = []
        self.female: list[bool] = []
        self.origin = self._person(None)

    def _person(self, parent: int | None) -> int:
        self.parent.append(parent)
        self.female.append(self.rng.random() < .5)
        return len(self.parent) - 1

    def up(self, p: int) -> int:
        if self.parent[p] is None:
            self.parent[p] = self._person(None)
        return self.parent[p]

    def blood(self, a: int, b: int) -> tuple[int, int]:
        """(ascent, descent) from `a` to `b` through their nearest common ancestor.

        Graph distance only. There is one root and everyone descends from it, so the climb
        from `b` always meets `a`'s line."""
        line, x, d = {}, a, 0
        while x is not None:
            line[x] = d
            x, d = self.parent[x], d + 1
        x, d = b, 0
        while x not in line:
            x, d = self.parent[x], d + 1
        return line[x], d

    def _realise(self, p: int, step: tuple[int, int]) -> tuple[int, tuple[int, ...]]:
        """A person standing in `step` to `p`, and who the step passes through. Anyone
        reached by going down or across is newly born, so only an ascent can meet someone
        the walk has already named."""
        if step == (1, 0):
            return self.up(p), ()
        if step == (2, 0):
            x = self.up(p)
            return self.up(x), (x,)
        if step == (0, 1):
            return self._person(p), ()
        if step == (0, 2):
            c = self._person(p)
            return self._person(c), (c,)
        if step == (1, 1):
            return self._person(self.up(p)), ()
        if step == (2, 1):
            x = self.up(p)
            return self._person(self.up(x)), (x,)
        s = self._person(self.up(p))
        return self._person(s), (s,)

    def walk(self, length: int, nameable: bool = False):
        """A walk from the origin that never meets the same person twice.

        Returns (chain, (ascent, descent), female). With `nameable`, every person the walk
        names is someone CLUTRR's twenty words can name relative to the origin -- the way
        CLUTRR's generator only emits questions it can answer -- so however long the chain,
        its answer is a word. Without it the walk wanders freely and the answer is usually
        a relative no word names, such as a thirtieth cousin."""
        p, seen, chain = self.origin, {self.origin}, []
        for _ in range(length):
            opts = []
            for step in self.STEPS:
                t, via = self._realise(p, step)
                if t in seen or any(v in seen for v in via):
                    continue
                if nameable and self.blood(self.origin, t) not in BLOOD:
                    continue
                opts.append((step, t, via))
            step, p, via = self.rng.choice(opts)
            chain.append(BLOOD[step][0 if self.female[p] else 1])
            seen.add(p)
            seen.update(via)
        return tuple(chain), self.blood(self.origin, p), self.female[p]

    def term(self, truth: tuple[int, int], female: bool) -> str | None:
        return BLOOD[truth][0 if female else 1] if truth in BLOOD else None


# ------------------------------------------------------------------ non-kinship worlds

class Compass:
    """Unit moves on a grid, and words for where two of them land -- one of which covers
    TWO places. The smallest world with a genuinely polysemous word and no people in it.

    The steps are `N`, `E`, `S`, `W` and `stay`. A composite is named by its displacement:
    `2N`, `NE`, `here` and so on. With `merge`, two of those names are replaced by one
    shared word, the way English lets one word cover two meanings: the word is a true union,
    right on either reading. The truth is vector addition over the steps, never a
    composition of names."""

    STEPS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0), "stay": (0, 0)}

    def __init__(self, merge: tuple[str, str] | None = ("NE", "SE"), word: str = "east-diagonal"):
        self.merge, self.word = merge, word

    def name(self, dx: int, dy: int) -> str | None:
        for s, v in self.STEPS.items():
            if v == (dx, dy):
                return s
        if abs(dx) + abs(dy) != 2:
            return None
        n = ("N" if dy > 0 else "S" if dy < 0 else "") + ("E" if dx > 0 else "W" if dx < 0 else "")
        n = n if abs(dx) != 2 and abs(dy) != 2 else f"2{n}"
        return self.word if self.merge and n in self.merge else n

    def truth(self, chain) -> str | None:
        dx = sum(self.STEPS[s][0] for s in chain)
        dy = sum(self.STEPS[s][1] for s in chain)
        return self.name(dx, dy)

    def products(self) -> list[tuple[tuple[str, str], str]]:
        return [((a, b), c) for a in self.STEPS for b in self.STEPS
                if (c := self.truth((a, b))) is not None]

    def chain(self, k: int, rng: random.Random):
        """A random walk of `k` steps whose endpoint has a name."""
        steps = list(self.STEPS)
        while True:
            c = tuple(rng.choice(steps) for _ in range(k))
            if (t := self.truth(c)) is not None:
                return c, t


class Flags:
    """A pointer on a line that also flips a mode switch: the smallest world with ROUTE
    state and no people in it.

    Each step moves the pointer and either SETS the switch, CLEARS it, or KEEPS whatever it
    was. The name of a composite is its total displacement and its net effect on the
    switch -- the last set or clear on the route, or keep if there was none. Displacement is
    additive, but the switch is not a function of any single step: `set` then `keep` is
    `set`, and the existing lanes place all three switch effects at one address.

    The truth is a direct left-to-right scan of the steps, never a composition of names,
    which is what makes it an independent check. Think of a path that enters and leaves a
    mounted filesystem: the mount it is under is decided by the last crossing, not the
    last step."""

    MARKS = ("keep", "set", "clear")

    def __init__(self, span: int = 2):
        self.span = span
        self.symbols = [self.name(d, m) for d in range(-span, span + 1) for m in self.MARKS]

    @staticmethod
    def name(d: int, mark: str) -> str:
        return f"{mark}{d:+d}"

    def truth(self, chain) -> str | None:
        d, mark = 0, "keep"
        for s in chain:
            m, step = s[:-2], int(s[-2:])
            d += step
            if m != "keep":
                mark = m
        return self.name(d, mark) if abs(d) <= self.span else None

    def products(self) -> list[tuple[tuple[str, str], str]]:
        return [((a, b), c) for a in self.symbols for b in self.symbols
                if (c := self.truth((a, b))) is not None]

    def chain(self, k: int, rng: random.Random):
        steps = [s for s in self.symbols if abs(int(s[-2:])) <= 1]
        while True:
            c = tuple(rng.choice(steps) for _ in range(k))
            if (g := self.truth(c)) is not None:
                return c, g



class Paths:
    """Filesystem path normalisation. The truth is `posixpath.normpath`, not an algebra.

    A relative path is a word: `..` steps out, a directory name steps in, and `x/..`
    cancels. That is the same monoid kinship turned out to be, wearing different clothes,
    and it has the property a family tree does not: a chain of ANY length still has a
    name, so depth can be pushed as far as anyone likes.

    With one directory name the world is exactly bicyclic and the solver should pin every
    answer. With two names it is richer than the cone can express -- `x/y` and `y/x` are
    different places with the same coordinates -- which is the interesting case: a sound
    system must widen its answer rather than pick."""

    def __init__(self, names: tuple[str, ...] = ("x",), up: int = 3, down: int = 3):
        self.names, self.up, self.down = names, up, down
        self.symbols = self._symbols()

    def _symbols(self) -> list[str]:
        words = [()]
        for _ in range(self.down):
            words += [w + (n,) for w in words if len(w) < self.down for n in self.names]
        out = set()
        for u in range(self.up + 1):
            for w in {tuple(x) for x in words}:
                out.add(posixpath.normpath("/".join([".."] * u + list(w)) or "."))
        return sorted(out)

    def compose(self, a: str, b: str) -> str:
        return posixpath.normpath(a + "/" + b)

    def products(self) -> list[tuple[tuple[str, str], str]]:
        """Every product of two symbols that lands back inside the vocabulary."""
        known = set(self.symbols)
        return [((a, b), self.compose(a, b)) for a in self.symbols for b in self.symbols
                if self.compose(a, b) in known]

    def table(self) -> dict[tuple[str, str], str]:
        """(state, step) -> state, for the SAMPLER only. The solver never sees it."""
        known = set(self.symbols)
        return {(a, b): c for a in self.symbols for b in self.symbols
                if (c := self.compose(a, b)) in known}

    def chain(self, k: int, rng: random.Random, table=None):
        """A chain of `k` steps and its true normalisation.

        Every step is chosen so the running path stays inside the vocabulary -- a
        traversal that stays within the directory tree, rather than one that wanders out
        of it and has to be thrown away. The answer is computed by walking, never by
        folding coordinates."""
        table = self.table() if table is None else table
        moves: dict[str, list[str]] = {}
        for (a, b) in table:
            moves.setdefault(a, []).append(b)
        here, steps = ".", []
        for _ in range(k):
            b = rng.choice(moves[here])
            steps.append(b)
            here = table[(here, b)]
        return tuple(steps), here


class Grid:
    """Displacement on a 2-D lattice. The truth is vector addition.

    Kinship needed one additive coordinate. This needs TWO, which is the point: the
    derivation returns a null space, and the dimension of that space is a fact about the
    world rather than a setting."""

    def __init__(self, span: int = 3):
        self.span = span
        self.symbols = [self.name(x, y) for x in range(-span, span + 1)
                        for y in range(-span, span + 1)]
        self.vec = {self.name(x, y): (x, y) for x in range(-span, span + 1)
                    for y in range(-span, span + 1)}

    @staticmethod
    def name(x: int, y: int) -> str:
        if x == y == 0:
            return "here"
        ew = f"{'E' if x > 0 else 'W'}{abs(x)}" if x else ""
        ns = f"{'N' if y > 0 else 'S'}{abs(y)}" if y else ""
        return ew + ns

    def compose(self, a: str, b: str) -> str | None:
        (x1, y1), (x2, y2) = self.vec[a], self.vec[b]
        if max(abs(x1 + x2), abs(y1 + y2)) > self.span:
            return None
        return self.name(x1 + x2, y1 + y2)

    def products(self) -> list[tuple[tuple[str, str], str]]:
        return [((a, b), c) for a in self.symbols for b in self.symbols
                if (c := self.compose(a, b)) is not None]

    def table(self) -> dict[tuple[str, str], str]:
        """(state, step) -> state, for the SAMPLER only."""
        return {(a, b): c for a in self.symbols for b in self.symbols
                if (c := self.compose(a, b)) is not None}

    def chain(self, k: int, rng: random.Random, table=None):
        """A chain of `k` steps that stays on the board, and where it ends."""
        table = self.table() if table is None else table
        moves: dict[str, list[str]] = {}
        for (a, b) in table:
            moves.setdefault(a, []).append(b)
        here, steps = "here", []
        for _ in range(k):
            b = rng.choice(moves[here])
            steps.append(b)
            here = table[(here, b)]
        return tuple(steps), here


# ------------------------------------------------------------------------- the runner

def _score(solver, rows):
    known = set(solver.system.symbols)
    rows = [(c, g) for c, g in rows if g in known and all(r in known for r in c)]
    n = len(rows) or 1
    return {"n": len(rows),
            "sound": sum(1 for c, g in rows if g in solver.solve(c)) / n,
            "top1": sum(1 for c, g in rows if solver.best(c) == g) / n,
            "set": sum(len(solver.solve(c)) for c, g in rows) / n}


def main(argv=None) -> int:
    """python -m bicyclic_kinship.worlds"""
    import argparse
    import time

    from .solver import Solver
    from . import clutrr

    ap = argparse.ArgumentParser(description="worlds beyond CLUTRR")
    ap.parse_args(argv)
    head = f"{'k':>7} {'chains':>7} {'sound':>7} {'top-1':>7} {'set':>6}"
    line = lambda k, r: (f"{k:7d} {r['n']:7d} {r['sound']:7.3f} {r['top1']:7.3f} "
                         f"{r['set']:6.2f}")

    print("=" * 72)
    print("1. A REAL FAMILY TREE, in CLUTRR's own twenty words")
    print("=" * 72)
    train_fam, test_fam = Family(generations=8, seed=5), Family(generations=8, seed=99)
    products = {}
    for _ in range(60000):
        got = train_fam.simple_walk(2)
        if got and (g := train_fam.term(got[0], got[2])) is not None:
            products[got[1]] = g
    tree_solver = Solver.train(sorted(products.items()))
    print(f"the tree: {len(train_fam)} people, {len(products)} observed products")
    print(f"  tree-derived  {tree_solver.system.describe()}")
    have_clutrr = clutrr.data_dir() is not None
    if have_clutrr:
        clutrr_solver = Solver.train(clutrr.load("train"))
        print(f"  CLUTRR-derived  {clutrr_solver.system.describe()}")
    print(f"\n{head}   trained on")
    for k in (2, 3, 5, 8):
        rows = []
        for _ in range(1500):
            got = test_fam.simple_walk(k)
            if got and (g := test_fam.term(got[0], got[2])) is not None:
                rows.append((got[1], g))
        if have_clutrr:
            print(line(k, _score(clutrr_solver, rows)) + "   CLUTRR")
        print(line(k, _score(tree_solver, rows)) + "   the tree itself")
    if have_clutrr:
        print("\n  the chain CLUTRR cannot teach:  mother o husband")
        print(f"    the tree says  'father'")
        print(f"    CLUTRR-derived laws admit  {clutrr_solver.solve(('mother', 'husband'))}"
              "   <- every CLUTRR chain ending in 'husband' answers 'son-in-law'")
        print(f"    tree-derived laws admit    {tree_solver.solve(('mother', 'husband'))}")

    print("\n" + "=" * 72)
    print("2. THE SAME TREE, with a vocabulary that keeps going: cousins, greats, removes")
    print("=" * 72)
    world = BloodWorld(Family(generations=8, children=(3, 4), seed=11), bound=7)
    pairs = world.products(20000)
    s = Solver.train(pairs)
    vocab = sorted(set(s.system.symbols))
    print(f"{len(pairs)} observed products over {len(vocab)} symbols, e.g. "
          f"{[v for v in vocab if 'cousin' in v][:2]}")
    print(f"  {s.system.describe()}")
    test = BloodWorld(Family(generations=8, children=(3, 4), seed=808), bound=7)
    print(f"\n{head}")
    for k in (2, 3, 5, 10, 20):
        rows = []
        for _ in range(1200):
            got = test.walk(k)
            if got and got[1] is not None:
                rows.append(got)
        r = _score(s, rows)
        print(line(k, r) if r["n"] else f"{k:7d} {0:7d}   the tree runs out: a walk this long "
              "must meet someone twice")

    print("\n" + "=" * 72)
    print("3. THE IN-LAW GAP: what CLUTRR could not teach, and what data cannot fix")
    print("=" * 72)
    w = KinWorld(Family(generations=7, children=(3, 4), seed=11), bound=3)
    pairs = w.products(30000)
    s = Solver.train(pairs)
    operands = sum(1 for c, _ in pairs if any(r.endswith("-in-law") for r in c))
    print(f"{len(pairs)} products over {len(s.system.symbols)} symbols, {operands} of them "
          f"with an in-law word as an OPERAND (CLUTRR has 0)")
    print(f"  {s.system.describe()}")
    print(f"  the address of 'father' admits {s.system.compose(['father'])}")
    test = KinWorld(Family(generations=7, children=(3, 4), seed=808), bound=3)
    print(f"\n{head}   in-law answers only")
    for k in (2, 3, 5):
        rows = [r for _ in range(1500) if (r := test.walk(k)) and r[1] is not None]
        known = set(s.system.symbols)
        rows = [(c, g) for c, g in rows if g in known and all(x in known for x in c)]
        if not rows:
            continue
        il = [(c, g) for c, g in rows if g.endswith("-in-law")]
        r = _score(s, rows)
        extra = (f"   {len(il):4d} chains, sound {_score(s, il)['sound']:.3f}, "
                 f"top-1 {_score(s, il)['top1']:.3f}") if il else ""
        print(line(k, r) + extra)

    print("\n" + "=" * 72)
    print("4. DEPTH: an endless family, walked to ten thousand steps")
    print("=" * 72)
    if have_clutrr:
        print("CLUTRR-derived laws (trained on k<=3), on walks that stay nameable")
        print(f"{head}   ms/answer")
        for k, n in ((10, 300), (100, 300), (1000, 100)):
            rows = []
            for i in range(n):
                w = Lineage(seed=k * 100000 + i)
                chain, truth, female = w.walk(k, nameable=True)
                rows.append((chain, w.term(truth, female)))
            t = time.time()
            r = _score(clutrr_solver, rows)
            print(line(k, r) + f"   {1000 * (time.time() - t) / (3 * n):8.2f}")
        anthro = next(i for i, c in enumerate(clutrr_solver.system.cone)
                      if c["father"] == (1, 0))
        print("\nfree walks: no word names the endpoint, so compare the ADDRESS itself")
        print(f"{'k':>7} {'walks':>7} {'exact':>7}   e.g. (ascent, descent)")
        for k, n in ((100, 100), (1000, 30), (10000, 5)):
            exact, eg = 0, None
            for i in range(n):
                w = Lineage(seed=7 + k + i)
                chain, truth, _ = w.walk(k)
                add, cone, _, _ = clutrr_solver.system.address(list(chain))
                exact += cone[anthro] == truth and add[0] == truth[0] - truth[1]
                eg = eg or truth
            print(f"{k:7d} {n:7d} {exact / n:7.3f}   {eg}")
    else:
        print("  (set CLUTRR_DIR to run the depth study)")

    print("\n" + "=" * 72)
    print("5. NOT KINSHIP AT ALL")
    print("=" * 72)
    for world, label, depths in (
            (Paths(("x",), 3, 3), "paths, one directory name -- the same monoid", (2, 100, 1000, 10000)),
            (Paths(("x", "y"), 2, 3), "paths, two names -- richer than the cone", (2, 100, 1000)),
            (Grid(3), "grid displacement -- additive dimension 2", (2, 100, 1000, 10000))):
        pairs = world.products()
        t = time.time()
        s = Solver.train(pairs)
        print(f"\n{label}\n  {len(world.symbols)} symbols, {len(pairs)} products, "
              f"derived in {time.time() - t:.1f}s")
        print(f"  {s.system.describe()}")
        rng = random.Random(0)
        tab = world.table()
        print(f"{head}")
        for k in depths:
            rows = [world.chain(k, rng, tab) for _ in range(200 if k <= 1000 else 10)]
            print(line(k, _score(s, rows)))
    return 0




class KinWorld:
    """The tree walked with MARRIAGE in it: blood terms, spouses, and in-laws.

    This exists to close the gap CLUTRR leaves open. There, all four in-law words are
    terminal -- they appear as answers and never as operands -- so nothing says how an
    in-law relation composes and any law about it is invented. Here they are steps like
    any other, and the products that were missing can simply be observed.

    Naming follows English: `b` is an in-law of `a` when `b` is a blood relative of `a`'s
    spouse, or the spouse of a blood relative of `a`, and the term is the blood term with
    "-in-law" after it. Both readings agree wherever both apply, and a person the two
    readings would name differently is dropped rather than resolved."""

    def __init__(self, family: Family, bound: int = 4, radius: int = 2):
        self.fam, self.bound, self.radius = family, bound, radius

    def term(self, a: int, b: int) -> str | None:
        f = self.fam
        if a == b or b is None:
            return None
        if f.spouse[a] == b:
            return "wife" if f.female[b] else "husband"
        got = f.blood(a, b)
        if got is not None:
            return kin_name(*got, f.female[b], self.bound)
        names = set()
        if f.spouse[a] is not None and (g := f.blood(f.spouse[a], b)) is not None:
            names.add(kin_name(*g, f.female[b], self.bound))
        if f.spouse[b] is not None and (g := f.blood(a, f.spouse[b])) is not None:
            names.add(kin_name(*g, f.female[b], self.bound))
        names.discard(None)
        if len(names) != 1:
            return None                       # unnameable, or named two ways: dropped
        return names.pop() + "-in-law"

    def neighbourhood(self, p: int):
        """People within `radius` edges of `p`, with the people passed through.

        Edges are the primitive ones -- parent, child, spouse -- so a step's intermediates
        are explicit and a walk can refuse to meet anyone twice."""
        f = self.fam
        seen = {p: frozenset()}
        frontier = [p]
        for _ in range(self.radius):
            nxt = []
            for x in frontier:
                via = seen[x] | {x}
                nbrs = f.parents(x) + list(f.kids[x])
                if f.spouse[x] is not None:
                    nbrs.append(f.spouse[x])
                for y in nbrs:
                    if y not in seen:
                        seen[y] = via - {p}
                        nxt.append(y)
            frontier = nxt
        seen.pop(p, None)
        return seen

    def steps(self, p: int, visited: frozenset[int] = frozenset()):
        out = []
        for target, via in self.neighbourhood(p).items():
            if target in visited or via & visited:
                continue
            name = self.term(p, target)
            if name is not None:
                out.append((name, target, frozenset(via) | {target}))
        return out

    def walk(self, length: int, tries: int = 12):
        f = self.fam
        for _ in range(tries):
            p = f.rng.randrange(len(f))
            origin, seen, chain = p, {p}, []
            while len(chain) < length:
                opts = self.steps(p, frozenset(seen))
                if not opts:
                    break
                word, nxt, via = f.rng.choice(opts)
                chain.append(word)
                seen |= via
                p = nxt
            if len(chain) == length:
                return tuple(chain), self.term(origin, p)
        return None

    def products(self, samples: int = 30000):
        out = {}
        for _ in range(samples):
            got = self.walk(2)
            if got and got[1] is not None:
                out[got[0]] = got[1]
        return sorted(out.items())

if __name__ == "__main__":
    raise SystemExit(main())

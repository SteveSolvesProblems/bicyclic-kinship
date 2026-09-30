"""The whole package, tested twice over.

TWO JOBS, and the order matters. The synthetic algebras come FIRST and are the guard: they
are the evidence that the solver derives its behaviour rather than knowing kinship. If any
domain knowledge ever leaks into the source, a stack algebra over `p`/`q`, a left-zero
semigroup and a structureless magma are what catch it -- none of them is kinship and each
has a different law.

The CLUTRR block then pins the reported numbers, so every figure in the README is a
regression test rather than a claim transcribed from a scratch script. It skips when the
data is absent; set `CLUTRR_DIR` to run it.
"""
import os
import random
from collections import defaultdict
from pathlib import Path

import pytest

from bicyclic_kinship import (AxisSystem, Solver, binary_table, cone_compose, cone_fold,
                              table_conflicts)
from bicyclic_kinship import clutrr
from bicyclic_kinship.benchmark import ceiling, evaluate, rows_for


# ====================================================================== the algebra decides

def stack_algebra():
    """A pure bicyclic monoid over opaque symbols: `q` pops, `p` pushes, and a pop
    followed by a push CANCELS. Not kinship, same law -- `q o p = 1` here is exactly
    `son o mother = self` there."""
    words = {"1": (0, 0), "p": (1, 0), "q": (0, 1), "pp": (2, 0), "qq": (0, 2),
             "pq": (1, 1), "ppq": (2, 1), "pqq": (1, 2)}
    out = []
    for a, ca in words.items():
        for b, cb in words.items():
            got = cone_compose(ca, cb)
            for name, c in words.items():
                if c == got:
                    out.append(((a, b), name))
    return out


def additive_ladder():
    """`s_i o s_j = s_{i+j}`: displacement with no cancellation and no classes."""
    return [((f"s{i}", f"s{j}"), f"s{i + j}")
            for i in range(4) for j in range(4) if i + j < 4]


def left_zero():
    """`a o b = a`. No additive law; a LEFT projection with every symbol its own class."""
    return [((a, b), a) for a in "wxyz" for b in "wxyz"]


def right_zero():
    """`a o b = b`. The mirror: a right projection and nothing else."""
    return [((a, b), b) for a in "wxyz" for b in "wxyz"]


def magma(withhold=()):
    """No law of any kind. A COMPLETE table needs no law, so the interesting case withholds
    a product and leaves a gap no derived law can fill."""
    syms = [f"m{i}" for i in range(4)]
    rng = random.Random(11)
    out = [((a, b), rng.choice(syms)) for a in syms for b in syms]
    return [p for p in out if p[0] not in withhold]


def test_the_stack_algebra_is_solved_exactly_and_extrapolates():
    """Cancellation, on symbols that are not kinship terms, at lengths never trained."""
    s = Solver.train(stack_algebra())
    assert s.system.cone
    assert s.solve(("q", "p")) == ["1"]                  # a pop then a push cancels
    assert s.solve(("q", "q", "p")) == ["q"]             # partial cancellation
    assert s.solve(("p", "q")) == ["pq"]                 # the other order does not
    assert s.solve(("q",) * 8 + ("p",) * 8) == ["1"]     # length 16 from length-2 products


def test_the_additive_ladder_is_solved_by_displacement_alone():
    s = Solver.train(additive_ladder())
    assert len(s.system.additive) == 1
    assert s.solve(("s1", "s1", "s1")) == ["s3"]


def test_projection_algebras_are_solved_on_both_sides():
    left = Solver.train(left_zero())
    assert left.solve(("w", "x", "y", "z")) == ["w"]
    right = Solver.train(right_zero())
    assert right.solve(("w", "x", "y", "z")) == ["z"]


def test_a_structureless_algebra_abstains_rather_than_guessing():
    """No law derivable and a product withheld: the honest answer is the whole vocabulary."""
    s = Solver.train(magma(withhold={("m0", "m1")}))
    assert s.system.describe().endswith("0 additive")
    assert s.solve(("m0", "m1")) == ["m0", "m1", "m2", "m3"]
    assert s.score(magma())["sound"] == 1.0


def test_preference_commits_where_the_laws_abstain():
    s = Solver.train(magma(withhold={("m0", "m1")}))
    chain = ("m0", "m1", "m2")
    assert len(s.solve(chain)) == 4                  # the laws know nothing
    assert s.best(chain) in s.solve(chain)           # the preference still commits


def test_a_wrapping_algebra_is_outside_these_families_and_abstains_soundly():
    """Z/5 is purely additive but WRAPS, so no rational invariant sees it and no cone law
    exists. The package is honest about that: it abstains, and stays sound.

    Recorded as a limitation, not hidden. A modular lane is what solves this, and it is not
    part of this release."""
    z5 = [((f"g{i}", f"g{j}"), f"g{(i + j) % 5}") for i in range(5) for j in range(5)]
    s = Solver.train(z5)
    assert not s.system.additive and not s.system.cone
    assert s.solve(("g1", "g2")) == sorted(s.system.symbols)
    assert s.score(z5)["sound"] == 1.0


def test_unknown_symbols_abstain_and_stay_sound():
    s = Solver.train(additive_ladder())
    assert s.solve(("s1", "nonesuch")) == sorted(s.system.symbols)


def test_contradictory_products_are_dropped_not_averaged():
    pairs = [(("a", "b"), "c"), (("a", "b"), "d"), (("b", "a"), "c")]
    assert ("a", "b") not in binary_table(pairs)
    assert table_conflicts(pairs) == [("a", "b")]


def test_a_false_law_is_discarded_by_validation():
    """A law fitted to the single-valued products of a many-valued algebra is simply false.
    Validation against longer training chains is what removes it -- without that, `solve`
    returns the EMPTY set on data the fit never saw."""
    pairs = additive_ladder() + [(("s1", "s1", "s1"), "s0")]   # contradicts g(a)+g(b)=g(c)
    kept = Solver.train(pairs)
    raw = Solver.train(pairs, validate=False)
    assert not kept.system.additive and raw.system.additive
    assert kept.solve(("s1", "s1", "s1")) != []


def test_the_cone_fold_is_associative():
    """The property the whole extrapolation claim rests on."""
    rng = random.Random(7)
    pts = [(rng.randrange(4), rng.randrange(4)) for _ in range(40)]
    for _ in range(200):
        a, b, c = (rng.choice(pts) for _ in range(3))
        assert cone_compose(cone_compose(a, b), c) == cone_compose(a, cone_compose(b, c))
    assert cone_fold([(1, 0), (1, 0), (0, 1)]) == (2, 1)   # up, up, down
    assert cone_fold([(0, 1), (1, 0)]) == (0, 0)           # down then up annihilates


def test_preference_never_leaves_what_the_laws_admit():
    """The safety property the split exists for, on an algebra with genuine ambiguity."""
    s = Solver.train(magma(withhold={("m0", "m1"), ("m2", "m3")}))
    for a in s.system.symbols:
        for b in s.system.symbols:
            assert s.best((a, b)) in s.solve((a, b))


# ====================================================================== CLUTRR regression

needs_clutrr = pytest.mark.skipif(clutrr.data_dir() is None,
                                  reason="CLUTRR absent; set CLUTRR_DIR")


@pytest.fixture(scope="module")
def root():
    return clutrr.data_dir()


@pytest.fixture(scope="module")
def train(root):
    return clutrr.load("train", root)


@pytest.fixture(scope="module")
def solver(train):
    return Solver.train(train)


@needs_clutrr
def test_clutrr_derivation_is_from_62_products_and_nothing_else(train, solver):
    """The entire input to the derivation: 62 observed products over 20 opaque symbols.
    Out of them come one displacement axis, four cone assignments and six terminal
    classes -- none of them supplied, and no table of longer chains anywhere."""
    assert len(binary_table([(tuple(c), a) for c, a in train])) == 62
    assert len(solver.system.symbols) == 20
    assert len(solver.system.additive) == 1
    assert len(solver.system.cone) == 4              # unique one depth shallower; see CONE_DEPTH_MARGIN
    assert len(set(solver.system.right.values())) == 6
    assert solver.system.left is None


@needs_clutrr
def test_clutrr_the_cone_coordinates_are_the_anthropological_ones(solver):
    """Read's algebraic account of American kinship, re-derived from opaque symbols.

    Ascent, descent and cancellation, with no name ever supplied: a parent is one step up,
    a child one step down, a spouse the identity, a sibling up-then-down."""
    cone = solver.system.cone[0]
    assert cone["father"] == cone["mother"] == (1, 0)
    assert cone["son"] == cone["daughter"] == (0, 1)
    assert cone["husband"] == cone["wife"] == (0, 0)
    assert cone["brother"] == cone["sister"] == (1, 1)
    assert cone["grandfather"] == (2, 0) and cone["granddaughter"] == (0, 2)
    assert cone_compose(cone["mother"], cone["son"]) == cone["brother"]


@needs_clutrr
def test_clutrr_soundness_is_total_on_every_instance(root, solver):
    """Trained once and never retrained: the gold answer is inside the admitted set on all
    2929 held-out stories of five independently generated instances."""
    total = 0
    for name, d in clutrr.instance_dirs(root):
        rows = rows_for(solver, clutrr.load("test", d))
        assert rows and all(g in solver.solve(c) for c, g in rows), name
        total += len(rows)
    assert total == 2929


@needs_clutrr
def test_clutrr_extrapolates_to_unseen_lengths_without_decay(root, solver):
    """Trained on k=2,3; measured on k=4..10. The reported 99.8%, pinned -- and FLAT: the
    hardest length is 3, inside the training distribution, because the genuinely ambiguous
    grandparent chains are short."""
    rows = defaultdict(list)
    for c, g in rows_for(solver, clutrr.load("test", root)):
        rows[len(c)].append((c, g))
    held = [(c, g) for k, v in rows.items() if k >= 4 for c, g in v]
    assert len(held) == 1003
    assert all(g in solver.solve(c) for c, g in held)
    top1 = sum(1 for c, g in held if solver.best(c) == g) / len(held)
    assert top1 >= 0.998
    for k in range(6, 11):                       # no decay with length
        assert all(solver.best(c) == g for c, g in rows[k])


@needs_clutrr
def test_clutrr_the_second_extrapolation_split(root, solver):
    """`gen_train234`: trained on k=2..4, measured on k=5..10 -- the split NCRL and EpiGNN
    report on. Trained on its own data it derives the SAME laws as `gen_train23`, is sound on
    every story, and scores 99.39% on the lengths it never saw. Every miss is a preference
    inside the admitted set; the `gen_train23` model, never retrained, makes none of them."""
    d = clutrr.train234_dir(root)
    if d is None:
        pytest.skip(f"{clutrr.TRAIN234} absent beside CLUTRR_DIR")
    own = Solver.train(clutrr.load("train", d))
    assert own.system.additive == solver.system.additive
    assert own.system.cone == solver.system.cone
    blocks = lambda part: {frozenset(s for s in part if part[s] == v) for v in part.values()}
    assert blocks(own.system.right) == blocks(solver.system.right)   # roots are arbitrary
    rows = rows_for(own, clutrr.load("test", d))
    assert len(rows) == 1048
    assert all(g in own.solve(c) for c, g in rows)
    held = [(c, g) for c, g in rows if len(c) >= 5]
    assert len(held) == 826
    assert sum(1 for c, g in held if own.best(c) == g) / len(held) >= 0.993    # 99.39%
    assert all(solver.best(c) == g for c, g in held)                           # 100%


@needs_clutrr
def test_clutrr_held_out_numbers(root, train, solver):
    """The reported result on the instance trained from. Bounds are measured values."""
    from bicyclic_kinship import ambiguous_chains
    test = clutrr.load("test", root)
    r = solver.score(test, ambiguous_chains(train, test))
    assert r["n"] == 1146
    assert r["sound"] == 1.0                 # what the laws ADMIT is never wrong
    assert r["unique"] >= 0.88               # 88.1%: a single admitted name
    assert r["top1"] >= 0.993                # 99.4%: the preferred name
    assert r["mean_set"] <= 1.13             # 1.12
    assert r["errors"] <= 2                  # PREFERENCE misses, inside the admitted set


@needs_clutrr
def test_clutrr_sits_at_the_ceiling(root, solver):
    """The ceiling is what ANY deterministic function of the chain could reach, given that
    the corpus answers some chains two ways. We match it exactly on four instances and are
    0.07% off over all five."""
    at_ceiling = num = den = 0
    for name, d in clutrr.instance_dirs(root):
        rows = rows_for(solver, clutrr.load("test", d))
        ours = sum(1 for c, g in rows if solver.best(c) == g)
        at_ceiling += ours == ceiling(rows)
        num += ours
        den += len(rows)
    assert at_ceiling >= 4
    r = evaluate(solver, [p for _, d in clutrr.instance_dirs(root)
                          for p in clutrr.load("test", d)])
    assert r["ceiling"] - num / den <= 0.001


@needs_clutrr
def test_clutrr_ambiguity_is_derived_not_declared(train, solver):
    """The one-to-many products are FOUND, not assumed: exactly the two addresses the laws
    place two names at are the two that training witnessed under two names -- and training
    only reveals it through chains longer than the pair itself."""
    got = sorted(tuple(sorted(solver.system.names_at(a)))
                 for a in solver.ambiguous_addresses())
    assert got == [("father", "father-in-law"), ("mother", "mother-in-law")]
    # the length-2 table sees only one of the two names, which is how the error arose
    assert binary_table([(tuple(c), a) for c, a in train])[("daughter", "grandmother")] \
        == "mother"


@needs_clutrr
def test_clutrr_every_ablation_is_still_sound(root, train):
    """Soundness is structural, not tuned: removing a family only LOOSENS the admitted set,
    so every subset of the derivation is still sound."""
    test = clutrr.load("test", root)
    for fam in [("additive", "right", "left"), ("additive", "cone"), ("cone",),
                ("cone", "right", "left")]:
        s = Solver.train(train, families=fam)
        rows = rows_for(s, test)
        assert all(g in s.solve(c) for c, g in rows), fam


@needs_clutrr
def test_clutrr_the_cone_lane_is_what_pays(root, train, solver):
    """Removing the cone lane costs six points of precision; removing the path rules costs
    under one. The geometry is the model; the rest is a tie-breaker."""
    test = clutrr.load("test", root)

    def top1(**kw):
        s = Solver.train(train, **kw)
        rows = rows_for(s, test)
        return sum(1 for c, g in rows if s.best(c) == g) / len(rows)

    full = top1()
    assert full - top1(families=("additive", "right", "left")) > 0.05
    assert 0 < full - top1(use_rules=False) < 0.01


@needs_clutrr
def test_clutrr_a_composite_with_no_name_returns_the_empty_set(solver):
    """The manifold is larger than the vocabulary, so a reachable address need not have a
    name. `husband o wife` is the identity and CLUTRR has no word for self; four ascents
    and three descents is a term the twenty symbols do not contain. Both come back EMPTY --
    "none of these", which is the honest answer, not the nearest one."""
    assert solver.solve(("husband", "wife")) == []
    assert solver.solve(("grandfather", "grandfather", "son", "son", "son")) == []


@needs_clutrr
def test_clutrr_disagreeing_routes_intersect_to_nothing(solver):
    """Soundness is conditional on the input being TRUE, and a false fact can invalidate an
    answer. What it cannot do is make the system confidently WRONG: two routes between the
    same pair of people intersect, and routes that disagree intersect to the empty set --
    a detected contradiction rather than a confident answer."""
    agree = set(solver.solve(("mother", "son"))) & set(solver.solve(("father", "son")))
    assert agree == {"brother"}
    assert not set(solver.solve(("mother", "son"))) & set(solver.solve(("father",)))


# ====================================================================== the README example

def custom_vocabulary():
    """The README's worked example: a vocabulary that is not CLUTRR's, from its products."""
    names = {"self": (0, 0), "parent": (1, 0), "child": (0, 1), "sibling": (1, 1),
             "grandparent": (2, 0), "grandchild": (0, 2), "pibling": (2, 1),
             "nibling": (1, 2)}
    return names, [((a, b), n) for a, ca in names.items() for b, cb in names.items()
                   for n, c in names.items() if c == cone_compose(ca, cb)]


def test_a_custom_vocabulary_solves_at_arbitrary_depth():
    """Every number quoted in the README's API example, pinned."""
    _, products = custom_vocabulary()
    assert len(products) == 42
    s = Solver.train(products)
    assert s.solve(("child", "parent", "sibling", "parent", "child")) == ["sibling"]
    assert s.solve(("child",) * 30 + ("parent",) * 30) == ["self"]
    assert s.solve(("child",) * 40 + ("parent",) * 39) == ["child"]
    assert s.solve(("child",) * 500 + ("parent",) * 500) == ["self"]


def test_the_coordinates_are_over_determined_by_the_products():
    """Withhold a fifth of the products and the answers do not move. That is the difference
    between a derived law and a table."""
    _, products = custom_vocabulary()
    part = [p for i, p in enumerate(products) if i % 5]
    s = Solver.train(part)
    assert s.solve(("child",) * 30 + ("parent",) * 30) == ["self"]
    assert s.solve(("child",) * 40 + ("parent",) * 39) == ["child"]


# ====================================================================== graphs and noise

@needs_clutrr
def test_the_derived_axes_are_identical_on_every_instance(root, solver):
    """The derivation is a property of kinship, not of one generated file: five
    independently generated datasets yield the same axis system, down to the same cone
    assignment -- and retraining on an instance's own data changes nothing."""
    from bicyclic_kinship import AxisSystem, triples_from
    for name, d in clutrr.instance_dirs(root):
        own_train = clutrr.load("train", d)
        s = AxisSystem.discover(triples_from([(tuple(c), a) for c, a in own_train]))
        assert len(s.additive) == 1 and len(s.cone) == 4, name
        assert len(set(s.right.values())) == 6, name
        assert any(c == solver.system.cone[0] for c in s.cone), name
        own = evaluate(Solver.train(own_train), clutrr.load("test", d))
        transferred = evaluate(solver, clutrr.load("test", d))
        assert own["top1"] == transferred["top1"], name


@needs_clutrr
def test_noise_true_but_irrelevant_facts_change_nothing(root, solver):
    """A distractor that does not connect the query's endpoints cannot create a false
    route, so a sound system is EXACTLY invariant to it -- not merely robust."""
    from bicyclic_kinship import noise
    stories = noise.usable(solver, clutrr.load_stories("test", root))
    vocab = sorted(solver.system.symbols)
    clean = noise.run(solver, stories, 0, "irrelevant", vocab)
    assert clean["wrong"] == 0.0
    for kind in ("irrelevant", "disconnected"):
        for n in (1, 5, 10):
            assert noise.run(solver, stories, n, kind, vocab) == clean, (kind, n)


@needs_clutrr
def test_noise_false_facts_are_flagged_never_answered_wrongly(root, solver):
    """Soundness is conditional on the input being true, so a false fact can invalidate an
    answer. What it cannot do is make the system confidently WRONG: routes that disagree
    intersect to nothing, and an empty intersection is a detected contradiction. Silently
    wrong answers stay at zero from one injected falsehood to ten."""
    from bicyclic_kinship import noise
    stories = noise.usable(solver, clutrr.load_stories("test", root))
    vocab = sorted(solver.system.symbols)
    for n in (1, 2, 5, 10):
        r = noise.run(solver, stories, n, "adversarial", vocab)
        assert r["wrong"] == 0.0, n                  # never confidently wrong
        assert r["flagged"] > 0.0, n                 # the contradiction is detected instead
    assert noise.run(solver, stories, 10, "adversarial", vocab)["flagged"] > 0.9


@needs_clutrr
def test_a_validation_selected_feature_loses_to_the_derived_preference(root, train, solver):
    """The trap CLUTRR is built to set, measured rather than argued.

    On the ambiguous addresses, "a spouse appears anywhere in the chain" beats the majority
    class on training AND on a validation split carved out of training -- and is worse than
    it held out. Not overfitting: distribution shift. A marriage crossing can be absorbed by
    a later descent, and chains of length 2-3 never show it. Recorded so the feature is not
    rediscovered and shipped on its validation number."""
    cone = solver.system.cone[0]
    spouse = {r for r in solver.system.symbols if cone[r] == (0, 0)}
    amb = {a for a in solver.ambiguous_addresses()
           if len(solver.system.names_at(a)) > 1}

    def on(pairs):
        return [(tuple(c), g) for c, g in pairs if solver.known(tuple(c))
                and g in set(solver.system.symbols)
                and solver.system.address(list(c)) in amb]

    def acc(rows, rule):
        return sum(1 for c, g in rows if rule(c) == g.endswith("-in-law")) / len(rows)

    fitted = sorted(set(on(train)))
    val = [p for i, p in enumerate(fitted) if not i % 5]
    held = on(clutrr.load("test", root))
    feature, prior = (lambda c: any(r in spouse for r in c)), (lambda c: False)
    assert acc(val, feature) > acc(val, prior)       # wins where it was selected
    assert acc(held, feature) < acc(held, prior)     # loses where it counts
    shipped = sum(1 for c, g in held if solver.best(c) == g) / len(held)
    assert shipped > acc(held, prior) > acc(held, feature)


# ====================================================================== beyond CLUTRR

from bicyclic_kinship.worlds import (BloodWorld, Family, Flags, Grid,      # noqa: E402
                                     KinWorld, Lineage, Paths)


def tree_products(fam: Family, samples: int = 40000):
    """Atomic two-step products the twenty words can name, observed by walking."""
    out = {}
    for _ in range(samples):
        got = fam.simple_walk(2)
        if got and (g := fam.term(got[0], got[2])) is not None:
            out[got[1]] = g
    return sorted(out.items())


@needs_clutrr
def test_a_real_family_tree_falsifies_a_law_clutrr_cannot(solver):
    """The sharpest limitation this project has measured, and it is about the DATA.

    Every chain in the whole CLUTRR corpus that ends in `husband` answers `son-in-law` --
    the only one that occurs is `daughter o husband`. So the terminal-class law
    `cls(a o b) = cls(b)` is true on every training chain, survives validation, and is
    FALSE in the world: your mother's husband is your father, and the laws admit nothing
    at all there. Soundness is audited against the data you have; a benchmark's coverage
    decides which laws are derivable, and this is what that costs."""
    fam = Family(generations=8, seed=99)
    assert solver.solve(("mother", "husband")) == []          # the truth is 'father'
    rows = []
    for _ in range(800):
        got = fam.simple_walk(2)
        if got and (g := fam.term(got[0], got[2])) is not None:
            rows.append((got[1], g))
    sound = sum(1 for c, g in rows if g in solver.solve(c)) / len(rows)
    assert 0.8 < sound < 0.95            # not sound off the benchmark's distribution


def test_the_same_derivation_on_the_tree_is_sound_and_exact():
    """And the repair is data, not cleverness. Given the tree's own products the six
    classes coarsen to four -- CLUTRR's extra two were an artifact of its coverage -- and
    on walks through a DIFFERENT tree every answer is right and unique."""
    s = Solver.train(tree_products(Family(generations=8, seed=5)))
    assert len(set(s.system.right.values())) == 4
    assert s.solve(("mother", "husband")) == ["father"]
    test = Family(generations=8, seed=99)
    for k in (2, 3, 5):
        rows = []
        for _ in range(600):
            got = test.simple_walk(k)
            if got and (g := test.term(got[0], got[2])) is not None:
                rows.append((got[1], g))
        assert rows and all(g in s.solve(c) for c, g in rows), k
        assert all(s.best(c) == g for c, g in rows), k


def test_an_open_vocabulary_of_cousins_and_greats_is_derived_the_same_way():
    """Nothing about the method is tied to twenty words. Naming (ascent, descent) by
    anthropology's rule gives cousins, removes and great-greats; the derivation takes 36
    symbols from 160 products and is exact on chains through an unseen tree."""
    world = BloodWorld(Family(generations=8, children=(3, 4), seed=11), bound=7)
    s = Solver.train(world.products(20000))
    assert len(s.system.symbols) > 30
    assert any("cousin" in v for v in s.system.symbols)
    assert any(v.startswith("great-") for v in s.system.symbols)
    test = BloodWorld(Family(generations=8, children=(3, 4), seed=808), bound=7)
    for k in (3, 5, 10):
        rows = [r for _ in range(900) if (r := test.walk(k)) and r[1] is not None]
        rows = [(c, g) for c, g in rows if g in set(s.system.symbols)
                and all(x in set(s.system.symbols) for x in c)]
        assert rows and all(g in s.solve(c) for c, g in rows), k
        assert all(s.best(c) == g for c, g in rows), k


def test_path_normalisation_is_the_same_monoid_and_folds_to_depth_10000():
    """Not kinship: `..` and a directory name, checked against posixpath.

    Two things a family tree cannot give. The world contains full cancellation, so the
    cone depth is its own reach: a world capped at k needs exactly k. And a chain of any
    length still has a name, so depth is limited by nothing at all."""
    import random
    world = Paths(("x",), 3, 3)
    s = Solver.train(world.products())
    assert len(s.system.cone) == 1                      # unique
    rng, tab = random.Random(0), world.table()
    for k in (2, 100, 1000, 10000):
        rows = [world.chain(k, rng, tab) for _ in range(20 if k > 1000 else 100)]
        assert all(s.solve(c) == [g] for c, g in rows), k


def test_a_world_richer_than_the_law_stays_sound_and_says_so():
    """With two directory names, `x/y` and `y/x` are different places with the same
    coordinates. The cone cannot separate them and does not pretend to: every answer set
    still contains the truth, and it has about six members instead of one. Sound where it
    cannot be precise is the whole design, tested where it costs something."""
    import random
    world = Paths(("x", "y"), 2, 3)
    s = Solver.train(world.products())
    rng, tab = random.Random(0), world.table()
    rows = [world.chain(k, rng, tab) for k in (2, 10, 100) for _ in range(60)]
    assert all(g in s.solve(c) for c, g in rows)
    assert 3 < sum(len(s.solve(c)) for c, _ in rows) / len(rows) < 9


def test_the_grid_derives_two_additive_axes_not_one():
    """CLUTRR has one additive coordinate, so one is all it can show. Displacement on a
    lattice needs two, and the dimension is read off the null space rather than set."""
    import random
    world = Grid(3)
    s = Solver.train(world.products())
    assert len(s.system.additive) == 2 and not s.system.cone
    rng, tab = random.Random(0), world.table()
    for k in (2, 100, 1000):
        rows = [world.chain(k, rng, tab) for _ in range(60)]
        assert all(s.solve(c) == [g] for c, g in rows), k


# ====================================================================== the obvious objections

@needs_clutrr
def test_soundness_is_not_the_trivial_abstention(solver, root):
    """"Returning every name is also 100% sound." It is -- and it scores 5% top-1 with
    twenty names per answer. The claim is soundness AT a mean admitted set of 1.12, with
    88% of chains pinned to a single name."""
    rows = rows_for(solver, clutrr.load("test", root))
    n = len(rows)
    assert sum(len(solver.solve(c)) for c, _ in rows) / n < 1.13
    assert sum(1 for c, _ in rows if len(solver.solve(c)) == 1) / n > 0.88
    assert sum(1 for c, g in rows if solver.best(c) == g) / n > 0.99


@needs_clutrr
def test_soundness_needs_enough_data_to_falsify_the_wrong_laws(train, root):
    """The uncomfortable one, and the same lesson as the family tree.

    Soundness is not a property of the method. It is a property of the method plus enough
    products to kill the laws that are false. With a quarter of the 62 products the
    derivation is confidently WRONG -- around 20% sound -- and it climbs back to 100% only
    once most of the table has been seen. Nothing warns you at the time; the laws look just
    as exact either way."""
    import random
    tab = sorted(binary_table([(tuple(c), a) for c, a in train]).items())
    test = clutrr.load("test", root)
    got = {}
    for frac in (0.25, 0.75, 1.0):
        rng = random.Random(0)
        keep = rng.sample(tab, max(2, int(len(tab) * frac)))
        s = Solver.train([((a, b), c) for (a, b), c in keep])
        rows = rows_for(s, test)
        got[frac] = sum(1 for c, g in rows if g in s.solve(c)) / max(len(rows), 1)
    assert got[0.25] < 0.5          # a quarter of the table: confidently wrong
    assert got[0.75] > 0.9          # most of it: nearly sound again
    assert got[1.0] == 1.0          # all of it: sound


@needs_clutrr
def test_exact_laws_are_brittle_to_label_noise_and_a_tolerance_repairs_them(train, root):
    """Derivation is exact, so one mislabelled answer in two hundred retires every law.

    The failure is SAFE -- the system abstains and stays sound -- and it is useless: top-1
    collapses to noise and the admitted set becomes the whole vocabulary. `tolerance`
    settles products by majority and keeps a law while the share of items it contradicts
    stays under the bound. It costs nothing on clean data and holds top-1 above 98% at ten
    percent corrupted labels."""
    import random
    test = clutrr.load("test", root)
    clean = Solver.train(train)
    assert evaluate(Solver.train(train, tolerance=0.1), test) == evaluate(clean, test)

    syms = sorted(clean.system.symbols)
    rng = random.Random(0)
    noisy = [(tuple(c), rng.choice(syms) if rng.random() < 0.01 else a) for c, a in train]

    exact = Solver.train(noisy)
    assert not exact.system.additive and not exact.system.cone   # every law retired
    rows = rows_for(exact, test)
    assert all(g in exact.solve(c) for c, g in rows)             # still sound, and useless
    assert sum(1 for c, g in rows if exact.best(c) == g) / len(rows) < 0.2

    tolerant = Solver.train(noisy, tolerance=0.1)
    rows = rows_for(tolerant, test)
    assert sum(1 for c, g in rows if tolerant.best(c) == g) / len(rows) > 0.98


@needs_clutrr
def test_label_noise_also_deletes_words(train):
    """The correction to the test above, pinned. A product whose witnesses disagree is
    dropped, and a word with no surviving product leaves the vocabulary -- so "still sound"
    holds only on the chains the model can still read. At one percent it loses words; at
    ten, every product disagrees and nothing is left. The tolerance keeps all twenty."""
    import random
    syms = sorted(Solver.train(train).system.symbols)

    def corrupt(rate):
        rng = random.Random(0)
        return [(tuple(c), rng.choice(syms) if rng.random() < rate else a) for c, a in train]

    assert len(Solver.train(corrupt(0.005)).system.symbols) == 20
    assert len(Solver.train(corrupt(0.01)).system.symbols) < 20
    assert len(Solver.train(corrupt(0.1)).system.symbols) == 0
    assert len(Solver.train(corrupt(0.1), tolerance=0.1).system.symbols) == 20


def test_an_algebra_outside_the_three_families_is_refused_soundly():
    """"It only works because kinship happens to be bicyclic." Correct -- and when a world
    is not, the system says so instead of fitting something.

    S3 is non-abelian; Z/4 wraps. Neither admits an additive axis over the rationals, a
    cone, or a terminal class. The derivation returns NO law, every answer is the whole
    vocabulary, and soundness is 100% because nothing was claimed."""
    import itertools
    for elems, mul, name in (
            (list(itertools.permutations(range(3))),
             lambda a, b: tuple(a[i] for i in b), lambda p: "p" + "".join(map(str, p))),
            (list(range(4)), lambda a, b: (a + b) % 4, lambda x: f"z{x}")):
        s = Solver.train([((name(a), name(b)), name(mul(a, b)))
                          for a in elems for b in elems])
        assert not s.system.additive and not s.system.cone and s.system.right is None
        assert s.solve((name(elems[1]), name(elems[1]))) == sorted(s.system.symbols)


@needs_clutrr
def test_the_family_tree_result_is_not_seed_luck(solver):
    """Five trees, five seeds, in the runner; two here to keep the suite quick."""
    for seed in (42, 1234):
        fam = Family(generations=8, seed=seed)
        rows, products = [], {}
        for _ in range(12000):
            got = fam.simple_walk(2)
            if got and (g := fam.term(got[0], got[2])) is not None:
                products[got[1]] = g
                if len(rows) < 500:
                    rows.append((got[1], g))
        own = Solver.train(sorted(products.items()))
        assert sum(1 for c, g in rows if g in solver.solve(c)) / len(rows) < 0.95
        assert all(g in own.solve(c) for c, g in rows)
        assert len(set(own.system.right.values())) == 4


# ====================================================================== closing two gaps

def test_the_sparse_elimination_equals_the_dense_reference():
    """The fast path must be the same mathematics, not a near-enough one.

    `_rref` is kept as the dense reference implementation and this checks the sparse
    incremental one against it -- on kinship, on a lattice, on paths and on an algebra with
    no additive law at all. Exact `Fraction` equality, not a tolerance."""
    from fractions import Fraction
    from bicyclic_kinship.derive import _null_space, additive_axes, triples_from

    def dense(triples):
        syms = sorted({s for t in triples for s in t})
        idx = {s: i for i, s in enumerate(syms)}
        rows = []
        for a, b, c in triples:
            row = [Fraction(0)] * len(syms)
            row[idx[a]] += 1
            row[idx[b]] += 1
            row[idx[c]] -= 1
            rows.append(row)
        return [dict(zip(syms, v)) for v in _null_space(rows, len(syms))]

    worlds = [Grid(3).products(), Paths(("x",), 3, 3).products(),
              Paths(("x", "y"), 2, 3).products(), additive_ladder(), magma(), left_zero()]
    for pairs in worlds:
        tri = triples_from([(tuple(c), a) for c, a in pairs])
        assert additive_axes(tri) == dense(tri)


def test_deriving_a_lattice_of_eight_hundred_symbols_is_seconds_not_minutes():
    """Rows from triples are three-sparse however large the vocabulary is, and there are
    far more rows than the rank can ever be. Reducing them as they arrive is what turns a
    225-symbol world from two minutes into under a second, and puts 841 symbols in reach."""
    import time
    world = Grid(10)
    t = time.time()
    s = Solver.train(world.products())
    assert time.time() - t < 30                      # measured at ~2.5s
    assert len(s.system.additive) == 2
    import random
    rng, tab = random.Random(0), world.table()
    assert all(s.solve(c) == [g] for c, g in
               (world.chain(50, rng, tab) for _ in range(20)))


def test_observing_in_law_products_closes_the_soundness_gap_but_not_the_precision_one():
    """The other half of the CLUTRR in-law gap, measured.

    CLUTRR never uses an in-law word as an operand, so their composition is invented. Walk
    a tree that has marriage in it and those products are simply observed -- and soundness
    comes back to 100% at every length.

    What does NOT come back is precision. `father` and `father-in-law` land at the SAME
    address: no law here distinguishes them, because in-law-ness is a property of the path
    that was walked -- whether it crossed a marriage, and whether a later descent absorbed
    the crossing -- rather than of the endpoints the coordinates describe. The path rules
    read exactly that feature, but a path rule may only narrow a set the laws admit, so it
    can express a preference and never a law. At length two the preference is right every
    time; by length five it is worse than a coin flip, because training favours the blood
    term."""
    world = KinWorld(Family(generations=6, children=(3, 4), seed=11), bound=3)
    s = Solver.train(world.products(9000))
    assert any(v.endswith("-in-law") for v in s.system.symbols)
    assert s.system.compose(["father"]) == ["father", "father-in-law"]

    test = KinWorld(Family(generations=6, children=(3, 4), seed=808), bound=3)
    known = set(s.system.symbols)
    for k in (2, 3):
        rows = [r for _ in range(500) if (r := test.walk(k)) and r[1] is not None]
        rows = [(c, g) for c, g in rows if g in known and all(x in known for x in c)]
        assert rows and all(g in s.solve(c) for c, g in rows), k    # sound again
    deep = [r for _ in range(500) if (r := test.walk(3)) and r[1] is not None]
    deep = [(c, g) for c, g in deep if g in known and all(x in known for x in c)
            and g.endswith("-in-law")]
    assert deep and sum(1 for c, g in deep if s.best(c) == g) / len(deep) < 0.95


def test_the_properties_the_derivation_rests_on():
    """Propositions 6-10 of the white paper, checked rather than asserted.

    The fold is a monoid (associative, with identity (0,0)); its shadow `d - u` is a
    homomorphism into the integers, which is what makes the cone search finite; translation
    by (k,k) is a gauge symmetry, which is what makes the solution set well defined; and
    the derived terminal partition refines every partition satisfying the same law, which
    is why the derived one dominates any hand-supplied version."""
    import itertools
    import random
    from bicyclic_kinship import projection_axis
    from bicyclic_kinship.derive import triples_from

    rng = random.Random(0)
    pts = [(rng.randrange(6), rng.randrange(6)) for _ in range(24)]

    for a, b, c in itertools.product(pts[:12], repeat=3):          # associativity
        assert cone_compose(cone_compose(a, b), c) == cone_compose(a, cone_compose(b, c))
    for a in pts:                                                  # identity
        assert cone_compose((0, 0), a) == a == cone_compose(a, (0, 0))
    for a, b in itertools.product(pts, repeat=2):                  # the linear shadow
        u, d = cone_compose(a, b)
        assert d - u == (a[1] - a[0]) + (b[1] - b[0])
        for k in (1, 4):                                           # gauge
            assert cone_compose((a[0] + k, a[1] + k), (b[0] + k, b[1] + k)) == (u + k, d + k)

    pairs = [(("f", "f"), "g"), (("f", "m"), "u"), (("m", "s"), "b"), (("f", "d"), "x")]
    triples = triples_from([(tuple(c), a) for c, a in pairs])
    derived = projection_axis(triples, "right")
    blocks = {}
    for sym, label in derived.items():
        blocks.setdefault(label, set()).add(sym)
    syms = sorted(derived)
    for assignment in itertools.product(range(3), repeat=len(syms)):
        other = dict(zip(syms, assignment))
        if all(other[c] == other[b] for _, b, c in triples):        # any valid partition
            for block in blocks.values():                          # is a coarsening
                assert len({other[x] for x in block}) == 1


def test_a_multi_valued_world_breaks_the_lane_NEGATIVE_RESULT():
    """Soundness has a precondition that Proposition 1 states but does not enforce:
    composition must be SINGLE-VALUED. `the` true product is doing work in that induction,
    and where a pair has two true composites there is nothing to apply validity to.

    Here a sibling's sibling may be yourself and a parent's child may be yourself, so two
    of the five observed products carry two answers. `binary_table` drops a conflicting
    product rather than guessing, which leaves the laws entirely unconstrained exactly
    there -- and since a symbol is admitted only where every law agrees, unanimity across
    them excludes the TRUE answer rather than a false one. The lane returns mostly EMPTY.

    This is the counterexample to "more laws can only narrow, never unsound". Narrowing is
    not free in either direction: fewer laws risk admitting a falsehood, more risk
    excluding the truth. CLUTRR escapes it not by luck but by a property that is checked --
    `table_conflicts` is zero on all five instances."""
    import random
    from bicyclic_kinship.derive import binary_table, table_conflicts, triples_from
    from bicyclic_kinship.worlds import Family

    fam = Family(generations=5)

    def sibs(p):
        f = fam.father[p]
        return [k for k in (fam.kids[f] if f is not None else []) if k != p]

    def step(p, w):
        if w == "parent":
            return [fam.father[p]] if fam.father[p] is not None else []
        return list(fam.kids[p]) if w == "child" else sibs(p)

    def name(p, q):
        if p == q:
            return "self"
        if q in sibs(p):
            return "sibling"
        if fam.father[p] == q:
            return "parent"
        return "child" if q in fam.kids[p] else None

    rng, people = random.Random(0), list(range(len(fam.father)))

    def sample(k, n):
        out = []
        for _ in range(n * 60):
            p = cur = rng.choice(people)
            chain = []
            for _ in range(k):
                nxt = step(cur, w := rng.choice(("parent", "child", "sibling")))
                if not nxt:
                    break
                cur, _ = rng.choice(nxt), chain.append(w)
            else:
                if (g := name(p, cur)):
                    out.append((tuple(chain), g))
            if len(out) >= n:
                break
        return out

    train, held = sample(2, 3000), [r for k in (3, 4, 5) for r in sample(k, 400)]
    assert len(table_conflicts(train)) >= 2          # the world is genuinely multi-valued
    assert len(binary_table(train)) < 5              # and the conflicting products are dropped

    def sound(s):
        return sum(1 for c, g in held if g in s.solve(c)) / len(held)

    # Unaudited, the lane is badly unsound: every law reproduces the products that survived,
    # they disagree freely on the two that did not, and unanimity excludes the TRUE answer.
    raw = Solver.train(train, validate=False)
    assert len(raw.system.cone) > 10
    assert sound(raw) < 0.5, sound(raw)

    # The audit is the defence, and here it is total -- every cone law contradicts some
    # training item, all of them are retired, and a system with no law admits everything.
    audited = Solver.train(train)
    assert len(audited.system.cone) == 0
    assert sound(audited) == 1.0

    # And the MARGIN is no help at all here: deeper only finds more laws that agree about
    # the products the data dropped, so soundness sits at the same figure however deep it
    # looks. This is the half of the pair that `validate` covers and the margin does not.
    assert sound(Solver.train(train, validate=False)) == sound(raw)


@needs_clutrr
def test_the_audit_does_not_catch_the_least_depth_law_which_is_why_the_margin_exists(train):
    """The other half of the pair, and the two do not overlap.

    `validate` retires a law the data REFUTES. The least-depth cone law is not refuted by
    anything: it says `husband o wife = wife`, which is false in the world -- it is SELF --
    but CLUTRR never records that product, so no training item of any length contradicts
    it. The audit keeps it, and the system asserts a name it should not have.

    So the audit is total on a world whose products conflict and blind on a product the
    corpus never shows, and the margin is the reverse. Neither subsumes the other, which is
    why both are here."""
    from bicyclic_kinship import AxisSystem, triples_from
    from bicyclic_kinship.cone import cone_axes
    from bicyclic_kinship.derive import additive_axes

    tri = triples_from(train)
    add = additive_axes(tri)
    least = next(b for b in range(1, 13) if cone_axes(tri, bound=b, limit=200, additive=add))
    assert least == 2

    one = cone_axes(tri, bound=least, limit=200, additive=add)
    assert len(one) == 1                                  # uniquely identified, and wrong
    syms = sorted({s for t_ in tri for s in t_})
    shallow = AxisSystem(add, None, None, syms, one)
    # both spouses sit on the identity, so the lane alone names both; the full system's
    # projection lane narrows it to ["wife"]. Either way it is a NAME, and the truth is SELF.
    assert shallow.compose(["husband", "wife"]) == ["husband", "wife"]

    # the audit does not save it -- nothing in the corpus contradicts the law
    assert len(shallow.validate(train).cone) == 1

    # one depth deeper the data admits four, they disagree there, and the answer is honest
    deeper = cone_axes(tri, bound=least + 1, limit=200, additive=add)
    assert len(deeper) == 4
    assert AxisSystem(add, None, None, syms, deeper).compose(["husband", "wife"]) == []


# ====================================================================== depth

def test_an_endless_family_is_answered_exactly_a_thousand_steps_deep():
    """A finite tree runs out of fresh people after about twenty steps, which is why depth
    stopped there. Grown on demand, it does not: derive from two-step products of the
    endless family and answer thousand-step chains whose truth is graph distance."""
    products = {}
    for i in range(4000):
        w = Lineage(seed=i)
        chain, truth, female = w.walk(2, nameable=True)
        products[chain] = w.term(truth, female)
    s = Solver.train(sorted(products.items()))
    for i in range(40):
        w = Lineage(seed=10 ** 6 + i)
        chain, truth, female = w.walk(1000, nameable=True)
        assert s.solve(chain) == [w.term(truth, female)]


@needs_clutrr
def test_clutrr_laws_hold_a_thousand_steps_deep(solver):
    """Derived from CLUTRR's chains of length two and three; asked about chains of a thousand
    in an endless family, every answer is the true one and the only one admitted."""
    for i in range(40):
        w = Lineage(seed=1000 * 100000 + i)
        chain, truth, female = w.walk(1000, nameable=True)
        assert solver.solve(chain) == [w.term(truth, female)]


@needs_clutrr
def test_clutrr_addresses_are_exact_where_no_word_exists(solver):
    """Ten thousand free steps land on a relative no word names, typically a few
    generations up and ten thousand down. The laws still place them EXACTLY: the derived cone
    coordinate is the true (ascent, descent), and the three assignments the corpus cannot
    rule out are the same coordinate translated by (1, 1)."""
    cones = solver.system.cone
    anthro = next(i for i, c in enumerate(cones) if c["father"] == (1, 0))
    for i in range(5):
        w = Lineage(seed=7 + 10000 + i)
        chain, (up, down), _ = w.walk(10000)
        add, cone, _, _ = solver.system.address(list(chain))
        assert cone[anthro] == (up, down)
        assert all(c == (up + 1, down + 1) for j, c in enumerate(cone) if j != anthro)
        assert add[0] == up - down
        assert solver.solve(chain) == []                   # no word, and no guess


# ====================================================================== route lane (opt-in)

def test_the_route_lane_derives_nothing_where_there_is_no_route_state():
    """The guard, again: on every world whose laws are already complete -- or which has no
    law at all -- the route lane must find nothing to add. A wrapping group is refused too:
    its only automaton has a state per symbol, which is the table, not a law."""
    z5 = [((f"g{i}", f"g{j}"), f"g{(i + j) % 5}") for i in range(5) for j in range(5)]
    z4 = [((f"z{i}", f"z{j}"), f"z{(i + j) % 4}") for i in range(4) for j in range(4)]
    for world in (stack_algebra(), additive_ladder(), Grid(3).products(), z5, z4,
                  left_zero(), right_zero(), magma()):
        for read in ("move", "move+label"):
            assert Solver.train(world, route=read).route is None


def test_the_route_lane_recovers_a_mode_switch_exactly():
    """A world with route state and no people: a pointer that sets, clears or keeps a
    switch. The existing lanes place all three switch effects at one address; the route
    lane derives exactly three states -- set, clear, keep -- and answers hundred-step chains
    uniquely. Reading the derived MOVE alone is not enough: the switch is invisible to it,
    so that variant stays sound and never becomes unique."""
    f = Flags()
    s = Solver.train(f.products(), route="move+label")
    assert s.route is not None and s.route.n_states == 3
    by_state = defaultdict(set)
    for sym, q in s.route.label.items():
        by_state[q].add(sym.rstrip("+-0123456789"))
    assert sorted(map(sorted, by_state.values())) == [["clear"], ["keep"], ["set"]]
    rng = random.Random(0)
    for _ in range(100):
        chain, truth = f.chain(100, rng)
        assert s.solve(chain) == [truth]
    blind = Solver.train(f.products(), route="move")
    for _ in range(50):
        chain, truth = f.chain(10, rng)
        got = blind.solve(chain)
        assert truth in got and len(got) > 1


def test_the_route_lane_finds_the_in_law_boundary_from_moves_alone():
    """On a family tree with marriage in it, reading only derived moves, the smallest
    route automaton has two states, and they split the vocabulary exactly at "-in-law" --
    a word the code never reads. It is sound and narrows the admitted set; it does not
    decide in-law answers on its own."""
    w = KinWorld(Family(generations=7, children=(3, 4), seed=11), bound=3)
    s = Solver.train(w.products(30000), route="move")
    assert s.route is not None and s.route.n_states == 2
    inlaw = {q for sym, q in s.route.label.items() if sym.endswith("-in-law")}
    blood = {q for sym, q in s.route.label.items() if not sym.endswith("-in-law")}
    assert len(inlaw) == 1 and len(blood) == 1 and inlaw != blood
    test = KinWorld(Family(generations=7, children=(3, 4), seed=808), bound=3)
    rows = [r for _ in range(400) if (r := test.walk(3)) and r[1] is not None]
    known = set(s.system.symbols)
    rows = [(c, g) for c, g in rows if g in known and all(x in known for x in c)]
    assert rows and all(g in s.solve(c) for c, g in rows)


@needs_clutrr
def test_clutrr_route_lane_is_retired_by_its_audit(train, root, solver):
    """On CLUTRR the search does find a two-state split -- the in-law words against the rest
    -- and 52 training chains contradict it, because the product `daughter o grandmother`
    is witnessed only as `mother`. The audit retires it and every answer is unchanged."""
    for read in ("move", "move+label"):
        s = Solver.train(train, route=read)
        assert s.route is None
        assert evaluate(s, clutrr.load("test", root)) == evaluate(solver, clutrr.load("test", root))


# ====================================================================== senses (opt-in)

from bicyclic_kinship.senses import SensedSolver                            # noqa: E402
from bicyclic_kinship.worlds import Compass                                 # noqa: E402


def test_senses_split_nothing_where_every_word_has_one_meaning():
    """The guard. On every world whose words each name one point, the split finds nothing
    and every answer is exactly the plain model's."""
    rng = random.Random(0)
    worlds = [stack_algebra(), additive_ladder(), left_zero(), right_zero(), magma(),
              [((f"g{i}", f"g{j}"), f"g{(i + j) % 5}") for i in range(5) for j in range(5)],
              [((f"g{i}", f"g{j}"), f"g{(i + j) % 4}") for i in range(4) for j in range(4)],
              Compass(merge=None).products(), Flags().products()]
    for w in (Paths(("x",), 3, 3), Grid(3)):
        worlds.append(w.products())
    for pairs in worlds:
        ss, s = SensedSolver.train(pairs), Solver.train(pairs)
        assert len(ss.senses) == 0
        assert all(ss.solve(c) == s.solve(c) for c, _ in pairs)
    lineage = {}
    for i in range(2000):
        w = Lineage(seed=i)
        chain, truth, female = w.walk(2, nameable=True)
        lineage[chain] = w.term(truth, female)
    assert len(SensedSolver.train(sorted(lineage.items())).senses) == 0


def test_senses_recover_a_word_with_two_meanings():
    """One word for both north-east and south-east costs the grid its second axis. The
    split recovers exactly the two meanings from the products alone -- the word's
    occurrences imply two different points -- and every long walk is answered uniquely."""
    w = Compass(merge=("NE", "SE"))
    assert len(Solver.train(w.products()).system.additive) == 1
    ss = SensedSolver.train(w.products())
    assert list(ss.senses.split) == ["east-diagonal"]
    got = sorted(sorted(g) for g in ss.senses.split["east-diagonal"])
    assert got == [[("E", "N"), ("N", "E")], [("E", "S"), ("S", "E")]]
    assert len(ss.system.additive) == 2
    rng = random.Random(1)
    for _ in range(200):
        chain, truth = w.chain(20, rng)
        assert ss.solve(chain) == [truth]


def test_senses_restore_the_laws_english_naming_destroys():
    """English says "first cousin once removed" in both directions and "cousin" for either
    sex, so one word names up to four places and every coordinate law is lost. Splitting
    by where each occurrence lands recovers them: one additive axis, the cone, and exact
    answers on walks through a tree the derivation never saw."""
    train = BloodWorld(Family(generations=8, children=(3, 4), seed=11), bound=7, naming="english")
    pairs = train.products(20000)
    assert len(Solver.train(pairs).system.additive) == 0
    ss = SensedSolver.train(pairs)
    assert {w: len(g) for w, g in ss.senses.split.items()} == \
        {"first cousin": 2, "first cousin 1x removed": 4}
    assert len(ss.system.additive) == 1 and ss.system.cone
    vocab = {x for c, a in pairs for x in (*c, a)}
    test = BloodWorld(Family(generations=8, children=(3, 4), seed=808), bound=7, naming="english")
    for k in (2, 3, 5):
        rows = [r for _ in range(400) if (r := test.walk(k)) and r[1] in vocab]
        assert rows and all(ss.solve(c) == [g] for c, g in rows)


def test_senses_find_the_double_meanings_of_in_law_words():
    """"Brother-in-law" is your sister's husband AND your spouse's brother. The split finds
    exactly that from the products, and with the class laws no longer collapsed by it, a
    tree with marriages is answered exactly -- on a tree the derivation never saw, where
    every answer is determined because a walk never passes anyone twice."""
    pairs = KinWorld(Family(generations=7, children=(3, 4), seed=11), bound=3).products(30000)
    ss = SensedSolver.train(pairs)
    senses = {frozenset(g) for g in ss.senses.split["brother-in-law"]}
    assert frozenset({("sister", "husband"), ("father", "son-in-law"),
                      ("mother", "son-in-law")}) in senses
    assert frozenset({("husband", "brother"), ("wife", "brother"),
                      ("father-in-law", "son"), ("mother-in-law", "son")}) in senses
    vocab = {x for c, a in pairs for x in (*c, a)}
    test = KinWorld(Family(generations=7, children=(3, 4), seed=808), bound=3)
    rows = [r for _ in range(600) if (r := test.walk(3)) and r[1] in vocab
            and all(x in vocab for x in r[0])]
    assert rows and all(ss.solve(c) == [g] for c, g in rows)


@needs_clutrr
def test_clutrr_senses_split_nothing(root, train, solver):
    """No CLUTRR word needs a second meaning, and every held-out answer is unchanged."""
    ss = SensedSolver.train(train)
    assert len(ss.senses) == 0
    rows = rows_for(solver, clutrr.load("test", root))
    assert all(ss.solve(c) == solver.solve(c) and ss.best(c) == solver.best(c)
               for c, _ in rows)


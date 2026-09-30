"""Stress tests: the measurements that try to break the model rather than score it.

    python -m bicyclic_kinship.stress [--root DIR] [--dense]

Six tables, each a question a reviewer would ask on reading the headline:

    families      what each law family is worth on `gen_train23`, one at a time
    evidence      how much of the 62-product table soundness needs -- five seeded random
                  subsets at each size
    labels        what corrupted training answers do, exact and with `tolerance=0.1`,
                  including the words the corruption deletes from the vocabulary
    trees         whether the family-tree result is one lucky seed
    outside       two algebras no family fits: the derivation must refuse, not fit
    cost          derivation time as the vocabulary grows; `--dense` also times the dense
                  reference elimination (slow: about two minutes on the largest world)

Every row is deterministic: the seeds are fixed, and nothing is averaged over runs.
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
import time

from . import clutrr
from .benchmark import rows_for
from .derive import _null_space, additive_axes, binary_table, triples_from
from .solver import Solver
from .worlds import Family, Grid, Paths


def score(solver: Solver, rows) -> dict:
    n = len(rows) or 1
    return {"n": len(rows),
            "sound": sum(1 for c, g in rows if g in solver.solve(c)) / n,
            "top1": sum(1 for c, g in rows if solver.best(c) == g) / n,
            "unique": sum(1 for c, _ in rows if len(solver.solve(c)) == 1) / n,
            "set": sum(len(solver.solve(c)) for c, _ in rows) / n}


def families(train, test):
    print("families -- gen_train23, held-out k = 2..10")
    print(f"  {'laws':24s} {'sound':>7s} {'top-1':>7s} {'unique':>7s} {'set':>6s}")
    for label, kw in (("all", {}),
                      ("without the cone", dict(families=("additive", "right", "left"))),
                      ("without the classes", dict(families=("additive", "cone"))),
                      ("without the additive", dict(families=("cone", "right", "left"))),
                      ("the cone alone", dict(families=("cone",))),
                      ("without the path rules", dict(use_rules=False))):
        s = Solver.train(train, **kw)
        r = score(s, rows_for(s, test))
        print(f"  {label:24s} {r['sound']:7.3f} {r['top1']:7.3f} {r['unique']:7.3f} "
              f"{r['set']:6.2f}")


def evidence(train, test):
    table = sorted(binary_table([(tuple(c), a) for c, a in train]).items())
    print(f"\nevidence -- products kept of {len(table)}, five seeded subsets each")
    print(f"  {'kept':>5s} {'sound':>7s} {'top-1':>7s} {'attempted':>10s}")
    for k in (15, 24, 31, 37, 46, 55, len(table)):
        got = []
        for seed in range(5):
            keep = random.Random(seed).sample(table, k)
            s = Solver.train([((a, b), c) for (a, b), c in keep])
            got.append(score(s, rows_for(s, test)))
        mean = {key: sum(r[key] for r in got) / 5 for key in ("sound", "top1", "n")}
        print(f"  {k:5d} {mean['sound']:7.3f} {mean['top1']:7.3f} {mean['n']:10.1f}")


def labels(train, test):
    clean = Solver.train(train)
    rows = rows_for(clean, test)
    syms = sorted(clean.system.symbols)
    print("\nlabels -- a share of training answers replaced by a random word")
    print("  a product whose witnesses disagree is dropped, so corruption also deletes words;")
    print("  'readable' is the share of held-out chains still inside the vocabulary, 'sound'")
    print("  is measured on those, and top-1 counts an unreadable chain as a miss")
    print(f"  {'corrupted':>9s}  {'exact: readable':>15s} {'sound':>6s} {'top-1':>6s} "
          f"{'set':>5s}   {'tolerant: readable':>18s} {'sound':>6s} {'top-1':>6s} {'set':>5s}")
    for rate in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        rng = random.Random(0)
        noisy = [(tuple(c), rng.choice(syms) if rng.random() < rate else a)
                 for c, a in train]
        line = f"  {rate:9.3f}"
        for s, width in ((Solver.train(noisy), 15), (Solver.train(noisy, tolerance=0.1), 18)):
            readable = rows_for(s, rows)
            r = score(s, readable)
            top1 = sum(1 for c, g in readable if s.best(c) == g) / len(rows)
            line += (f"  {len(readable) / len(rows):{width}.3f} "
                     + (f"{r['sound']:6.3f}" if readable else f"{'--':>6s}")
                     + f" {top1:6.3f} " + (f"{r['set']:5.1f}" if readable else f"{'--':>5s}")
                     + " ")
        print(line.rstrip())


def trees(clutrr_solver):
    print("\ntrees -- five family trees, 500 two-step walks each")
    print(f"  {'seed':>5s} {'people':>7s} {'CLUTRR-trained sound':>21s} "
          f"{'own sound':>10s} {'own top-1':>10s} {'classes':>8s}")
    for seed in (5, 42, 99, 1234, 7777):
        fam = Family(generations=8, seed=seed)
        rows, products = [], {}
        for _ in range(12000):
            got = fam.simple_walk(2)
            if got and (g := fam.term(got[0], got[2])) is not None:
                products[got[1]] = g
                if len(rows) < 500:
                    rows.append((got[1], g))
        own = Solver.train(sorted(products.items()))
        c, o = score(clutrr_solver, rows), score(own, rows)
        print(f"  {seed:5d} {len(fam.female):7d} {c['sound']:21.3f} {o['sound']:10.3f} "
              f"{o['top1']:10.3f} {len(set(own.system.right.values())):8d}")


def outside():
    print("\noutside -- algebras no family fits")
    for label, elems, mul, name in (
            ("S3, non-abelian", list(itertools.permutations(range(3))),
             lambda a, b: tuple(a[i] for i in b), lambda p: "p" + "".join(map(str, p))),
            ("Z/4, wraps", list(range(4)), lambda a, b: (a + b) % 4, lambda x: f"z{x}")):
        pairs = [((name(a), name(b)), name(mul(a, b))) for a in elems for b in elems]
        s = Solver.train(pairs)
        rng = random.Random(0)
        rows = []
        for _ in range(200):
            w = [rng.choice(elems) for _ in range(rng.randint(2, 6))]
            v = w[0]
            for x in w[1:]:
                v = mul(v, x)
            rows.append((tuple(name(x) for x in w), name(v)))
        r = score(s, rows)
        print(f"  {label:16s} {s.system.describe()}")
        print(f"  {'':16s} sound {r['sound']:.3f}  top-1 {r['top1']:.3f}  "
              f"set {r['set']:.2f} of {len(elems)}")


def cost(dense: bool):
    print("\ncost -- derivation time")
    print("  'derive' is the whole derivation; the two kernel columns time the additive")
    print("  null space alone, sparse incremental against the dense reference")
    print(f"  {'world':18s} {'symbols':>7s} {'products':>9s} {'derive':>8s} {'kernel':>8s}"
          + (f" {'dense kernel':>13s}" if dense else ""))
    for label, world in (("paths, two names", Paths(("x", "y"), 3, 3)),
                         ("grid, span 5", Grid(5)), ("grid, span 7", Grid(7)),
                         ("grid, span 10", Grid(10)), ("grid, span 14", Grid(14))):
        pairs = world.products()
        syms = {x for c, a in pairs for x in (*c, a)}
        t = time.perf_counter()
        Solver.train(pairs)
        line = f"  {label:18s} {len(syms):7d} {len(pairs):9d} {time.perf_counter() - t:7.2f}s"
        tri = triples_from([(tuple(c), a) for c, a in pairs])
        t = time.perf_counter()
        additive_axes(tri)
        line += f" {time.perf_counter() - t:7.2f}s"
        if dense and len(syms) <= 225:
            from fractions import Fraction
            idx = {s: i for i, s in enumerate(sorted(syms))}
            rows = []
            for a, b, c in tri:
                row = [Fraction(0)] * len(idx)
                row[idx[a]] += 1
                row[idx[b]] += 1
                row[idx[c]] -= 1
                rows.append(row)
            t = time.perf_counter()
            _null_space(rows, len(idx))
            line += f" {time.perf_counter() - t:12.2f}s"
        print(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None,
                    help="a CLUTRR instance directory (default: $CLUTRR_DIR)")
    ap.add_argument("--dense", action="store_true",
                    help="also time the dense reference elimination")
    args = ap.parse_args(argv)

    root = args.root or clutrr.data_dir()
    if root is None:
        print("No CLUTRR data. Set CLUTRR_DIR to an instance directory.", file=sys.stderr)
        return 2
    train, test = clutrr.load("train", root), clutrr.load("test", root)
    families(train, test)
    evidence(train, test)
    labels(train, test)
    trees(Solver.train(train))
    outside()
    cost(args.dense)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

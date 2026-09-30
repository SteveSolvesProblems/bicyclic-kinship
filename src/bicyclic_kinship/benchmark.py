"""The CLUTRR evaluation, as a runnable script.

    python -m bicyclic_kinship.benchmark [--root DIR] [--no-ablation]

Trains ONCE on the `gen_train23` training split -- chains of length 2 and 3 only -- and
evaluates without retraining on the held-out splits of every CLUTRR instance found beside
it. Nothing is refitted per instance and no instance's test data is ever seen.

Four tables:

    per instance   soundness, top-1, mean admitted set, and the CEILING -- what any
                   deterministic function of the relation chain could score, given that the
                   corpus itself answers some chains two ways.
    per length     the extrapolation curve, k = 2 .. 10, on the instance trained from.
    gen_train234   the second extrapolation split, k = 5 .. 10, trained on its own k = 2 .. 4
                   -- the protocol the published table uses. Printed when it is present.
    ablation       what each law family and the path rules are worth.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from . import clutrr
from .derive import ambiguous_chains, binary_table
from .solver import Solver
from .system import AxisSystem


def rows_for(solver: Solver, pairs) -> list[tuple[tuple[str, ...], str]]:
    """Items whose chain and answer are entirely within the trained vocabulary."""
    known = set(solver.system.symbols)
    return [(tuple(c), a) for c, a in pairs
            if c and a in known and all(r in known for r in c)]


def ceiling(rows) -> int:
    """The best any function of the chain alone could do on these rows."""
    gold: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c, g in rows:
        gold[c][g] += 1
    return sum(max(v.values()) for v in gold.values())


def evaluate(solver: Solver, pairs) -> dict:
    rows = rows_for(solver, pairs)
    n = len(rows) or 1
    sound = sum(1 for c, g in rows if g in solver.solve(c))
    top1 = sum(1 for c, g in rows if solver.best(c) == g)
    size = sum(len(solver.solve(c)) for c, _ in rows)
    return {"n": len(rows), "sound": sound / n, "top1": top1 / n,
            "mean_set": size / n, "ceiling": ceiling(rows) / n}


def by_length(solver: Solver, pairs) -> dict[int, dict]:
    out: dict[int, list] = defaultdict(list)
    for c, g in rows_for(solver, pairs):
        out[len(c)].append((c, g))
    return {k: evaluate(solver, v) for k, v in sorted(out.items())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None,
                    help="a CLUTRR instance directory (default: $CLUTRR_DIR)")
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args(argv)

    root = args.root or clutrr.data_dir()
    if root is None:
        print("No CLUTRR data. Set CLUTRR_DIR to an instance directory, e.g.\n"
              "  export CLUTRR_DIR=.../clutrr/gen_train23_test2to10", file=sys.stderr)
        return 2

    train = clutrr.load("train", root)
    solver = Solver.train(train)
    triples = binary_table([(tuple(c), a) for c, a in train])
    print(f"trained on {len(train)} chains of length 2-3 from {Path(root).name}")
    print(f"derived from {len(triples)} observed products over "
          f"{len(solver.system.symbols)} symbols")
    print(f"  {solver.system.describe()}\n")

    instances = clutrr.instance_dirs(root) or [(Path(root).name, Path(root))]
    print(f"{'instance':36s} {'n':>5s} {'sound':>8s} {'top-1':>8s} {'set':>6s} {'ceiling':>8s}")
    print("-" * 76)
    tot = defaultdict(float)
    for name, d in instances:
        r = evaluate(solver, clutrr.load("test", d))
        print(f"{name:36s} {r['n']:5d} {r['sound']:8.4f} {r['top1']:8.4f} "
              f"{r['mean_set']:6.3f} {r['ceiling']:8.4f}")
        tot["n"] += r["n"]
        for k in ("sound", "top1", "mean_set", "ceiling"):
            tot[k] += r[k] * r["n"]
    n = tot["n"]
    print("-" * 76)
    print(f"{'TOTAL':36s} {int(n):5d} {tot['sound']/n:8.4f} {tot['top1']/n:8.4f} "
          f"{tot['mean_set']/n:6.3f} {tot['ceiling']/n:8.4f}")

    test = clutrr.load("test", root)
    lengths = by_length(solver, test)
    print(f"\nextrapolation on {Path(root).name} (trained on k=2,3 only)")
    ks = sorted(lengths)
    print("k      " + "".join(f"{k:>8d}" for k in ks))
    print("n      " + "".join(f"{lengths[k]['n']:>8d}" for k in ks))
    print("sound  " + "".join(f"{lengths[k]['sound']:>8.3f}" for k in ks))
    print("top-1  " + "".join(f"{lengths[k]['top1']:>8.3f}" for k in ks))
    held = [v for k, v in lengths.items() if k >= 4]
    hn = sum(v["n"] for v in held)
    print(f"\nk>=4 (unseen lengths): n={hn}  "
          f"sound={sum(v['sound']*v['n'] for v in held)/hn:.4f}  "
          f"top-1={sum(v['top1']*v['n'] for v in held)/hn:.4f}")

    d234 = clutrr.train234_dir(root)
    if d234 is not None:
        own = Solver.train(clutrr.load("train", d234))
        test234 = clutrr.load("test", d234)
        lengths = by_length(own, test234)
        print(f"\nextrapolation on {d234.name} (trained on its own k=2,3,4)")
        print(f"  {own.system.describe()}")
        ks = sorted(lengths)
        print("k      " + "".join(f"{k:>8d}" for k in ks))
        print("n      " + "".join(f"{lengths[k]['n']:>8d}" for k in ks))
        print("sound  " + "".join(f"{lengths[k]['sound']:>8.3f}" for k in ks))
        print("top-1  " + "".join(f"{lengths[k]['top1']:>8.3f}" for k in ks))
        held = [(c, g) for c, g in rows_for(own, test234) if len(c) >= 5]
        for label, s in (("own model", own), ("gen_train23 model", solver)):
            r = evaluate(s, held)
            print(f"k>=5 (unseen lengths), {label:17s}: n={r['n']}  "
                  f"sound={r['sound']:.4f}  top-1={r['top1']:.4f}")

    if not args.no_ablation:
        print("\nablation (all five instances, top-1)")
        configs = [
            ("everything", {}),
            ("without the cone lane", dict(families=("additive", "right", "left"))),
            ("without the projection lane", dict(families=("additive", "cone"))),
            ("cone lane alone", dict(families=("cone",))),
            ("without the path rules", dict(use_rules=False)),
        ]
        for label, kw in configs:
            s = Solver.train(train, **kw)
            num = den = snd = 0
            for _, d in instances:
                rows = rows_for(s, clutrr.load("test", d))
                num += sum(1 for c, g in rows if s.best(c) == g)
                snd += sum(1 for c, g in rows if g in s.solve(c))
                den += len(rows)
            print(f"  {label:30s} top-1 {num/den:.4f}   sound {snd/den:.4f}")

    r = Solver.train(train).score(test, ambiguous_chains(train, test))
    print(f"\npreference misses outside the corpus's own ambiguity: {r['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

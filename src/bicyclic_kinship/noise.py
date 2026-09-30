"""Noise: what happens when the graph carries facts the question did not ask for.

CLUTRR's own noise conditions live in the prose -- the extra facts are in `story_edges`
but their relation labels are only in the text -- so they are reconstructed here on the
symbolic side, which is stricter about what counts as an answer.

A story is a GRAPH. Rather than being handed the chain, the solver enumerates every simple
route from the query's source to its target, composes each one, and INTERSECTS the admitted
sets. Intersection is the sound combination: extra true facts can only narrow an answer,
never widen it.

Three conditions:

    irrelevant     a true edge from a story entity to a fresh one
    disconnected   a true edge in a component the query cannot reach
    adversarial    a FALSE edge between two story entities -- which CLUTRR does not test

The first two cannot create a route between the endpoints, so a sound system must be
EXACTLY invariant to them, not merely robust. The third can, and the question it answers is
the one that matters for a system that claims soundness: when the input is false, does the
system go quietly wrong, or does it notice? Two routes that disagree intersect to nothing,
and an empty intersection is a detected contradiction.

    python -m bicyclic_kinship.noise
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from . import clutrr
from .solver import Solver


def routes(adj, src, dst, limit: int = 14) -> list[tuple[str, ...]]:
    """Every simple route from `src` to `dst`, as relation chains."""
    out, stack = [], [(src, (), frozenset({src}))]
    while stack:
        node, path, seen = stack.pop()
        if node == dst and path:
            out.append(path)
            continue
        if len(path) >= limit:
            continue
        for b, r in adj.get(node, ()):
            if b not in seen:
                stack.append((b, path + (r,), seen | {b}))
    return out


def noisy_graph(story: clutrr.Story, n: int, kind: str, rng: random.Random, vocab):
    """The story's graph with `n` extra edges of the given kind."""
    adj = defaultdict(list)
    for (a, b), r in zip(story.edges, story.chain):
        adj[a].append((b, r))
    nodes = sorted({x for e in story.edges for x in e})
    nxt, fresh = max(nodes) + 1, []
    for _ in range(n):
        if kind == "disconnected":
            a = rng.choice(fresh) if fresh and rng.random() < .5 else nxt
            nxt += 1
            adj[a].append((nxt, rng.choice(vocab)))
            fresh += [a, nxt]
            nxt += 1
        elif kind == "irrelevant":
            adj[rng.choice(nodes)].append((nxt, rng.choice(vocab)))
            fresh.append(nxt)
            nxt += 1
        else:                                   # adversarial: a FALSE edge, both ends real
            adj[rng.choice(nodes)].append((rng.choice(nodes), rng.choice(vocab)))
    return adj


def run(solver: Solver, stories, n: int, kind: str, vocab, seed: int = 0) -> dict:
    """Compose every route, intersect, and count what came out."""
    rng = random.Random(seed)
    total = sound = flagged = wrong = 0
    for st in stories:
        found = routes(noisy_graph(st, n, kind, rng, vocab), *st.query)
        if not found:
            continue
        total += 1
        admitted = set.intersection(*[set(solver.solve(p)) for p in found])
        if not admitted:
            flagged += 1
        else:
            sound += st.answer in admitted
            wrong += st.answer not in admitted
    d = total or 1
    return {"n": total, "sound": sound / d, "flagged": flagged / d, "wrong": wrong / d}


def usable(solver: Solver, stories) -> list[clutrr.Story]:
    """Stories whose chain and answer are inside the trained vocabulary."""
    known = set(solver.system.symbols)
    return [s for s in stories
            if s.chain and s.answer in known and all(r in known for r in s.chain)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 5, 10])
    args = ap.parse_args(argv)

    root = args.root or clutrr.data_dir()
    if root is None:
        print("No CLUTRR data. Set CLUTRR_DIR.", file=sys.stderr)
        return 2

    solver = Solver.train(clutrr.load("train", root))
    stories = usable(solver, clutrr.load_stories("test", root))
    vocab = sorted(solver.system.symbols)
    print(f"{len(stories)} stories, routes found in the graph rather than given\n")
    head = "".join(f"{k:>9d}" for k in args.levels)
    for kind in ("irrelevant", "disconnected", "adversarial"):
        print(f"{kind} facts per story{head}")
        rows = [run(solver, stories, k, kind, vocab) for k in args.levels]
        print("  answered and sound " + "".join(f"{r['sound']:9.3f}" for r in rows))
        print("  contradiction flag " + "".join(f"{r['flagged']:9.3f}" for r in rows))
        print("  SILENTLY WRONG     " + "".join(f"{r['wrong']:9.3f}" for r in rows))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

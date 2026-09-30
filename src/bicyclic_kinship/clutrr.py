"""CLUTRR loading. PERFECT EXTRACTION: the symbolic form is used, prose is never parsed.

CLUTRR ships each story as templated surface text AND as the structured relation chain that
generated it. `edge_types` gives the chain and `target_text` the composed relation, so a
parsing failure cannot contaminate a reasoning measurement -- there is no parsing.

This is the same input the symbolic baselines consume (R5 reads triples; Edge Transformer
uses "the noiseless graph-based version"), which is what makes the comparison with them a
like-for-like one.

The data lives outside this package: the CSV files behind the HuggingFace release
(CLUTRR/v1), one directory per instance, which the README shows how to fetch. Point
`CLUTRR_DIR` at one instance directory, or pass `root` explicitly.
"""
from __future__ import annotations

import ast
import csv
import os
from pathlib import Path
from typing import NamedTuple

csv.field_size_limit(10 ** 7)

#: The five independently generated instances the benchmark reports.
INSTANCES = (
    "gen_train23_test2to10",
    "rob_train_clean_23_test_all_23",
    "rob_train_disc_23_test_all_23",
    "rob_train_irr_23_test_all_23",
    "rob_train_sup_23_test_all_23",
)

#: The sixth instance, and the second extrapolation split: trained on k=2..4, tested on
#: k=5..10. It is the split the recent published table uses (NCRL, EpiGNN), so it is scored
#: on that protocol -- trained on its OWN training split -- rather than folded into the five
#: above, which share one model trained on `gen_train23`.
TRAIN234 = "gen_train234_test2to10"


def data_dir() -> Path | None:
    """The instance directory `CLUTRR_DIR` points at, or None."""
    d = os.environ.get("CLUTRR_DIR")
    if not d:
        return None
    p = Path(d)
    return p if p.is_dir() else None


def load(split: str, root: Path | None = None) -> list[tuple[tuple[str, ...], str]]:
    """`(relation chain, target relation)` pairs. No text, no parsing."""
    root = root or data_dir()
    if root is None:
        raise FileNotFoundError("CLUTRR data not found; set CLUTRR_DIR")
    out: list[tuple[tuple[str, ...], str]] = []
    with open(Path(root) / f"{split}.csv") as fh:
        for row in csv.DictReader(fh):
            try:
                chain = tuple(ast.literal_eval(row["edge_types"]))
            except (ValueError, SyntaxError):
                continue
            out.append((chain, row["target_text"].strip()))
    return out


def instance_dirs(root: Path | None = None) -> list[tuple[str, Path]]:
    """Every instance in `INSTANCES` present beside the configured one."""
    base = root or data_dir()
    if base is None:
        return []
    parent = Path(base).parent
    return [(d, parent / d) for d in INSTANCES if (parent / d).is_dir()]


def train234_dir(root: Path | None = None) -> Path | None:
    """The `gen_train234` instance beside the configured one, or None."""
    base = root or data_dir()
    if base is None:
        return None
    d = Path(base).parent / TRAIN234
    return d if d.is_dir() else None


class Story(NamedTuple):
    """One item WITH its graph. `chain[i]` is the relation from the entity at `edges[i][0]`
    to the one at `edges[i][1]`, and the question is the relation from `query[0]` to
    `query[1]`.

    The wider view exists for one experiment: a story is a graph, so a solver can be asked
    to FIND the route rather than be handed it, and noise can be added to the graph. Every
    field is read from a structured column; nothing here parses prose either."""

    chain: tuple[str, ...]
    answer: str
    edges: tuple[tuple[int, int], ...]
    query: tuple[int, int]


def load_stories(split: str, root: Path | None = None) -> list[Story]:
    """The wide view: chain, answer, story graph, query edge.

    A row is skipped only when a structured column will not parse, never partially read."""
    root = root or data_dir()
    if root is None:
        raise FileNotFoundError("CLUTRR data not found; set CLUTRR_DIR")
    out: list[Story] = []
    with open(Path(root) / f"{split}.csv") as fh:
        for row in csv.DictReader(fh):
            try:
                chain = tuple(ast.literal_eval(row["edge_types"]))
                edges = tuple(tuple(e) for e in ast.literal_eval(row["story_edges"]))
                qedge = tuple(ast.literal_eval(row["query_edge"]))
            except (ValueError, SyntaxError, TypeError):
                continue
            if len(edges) != len(chain) or len(qedge) != 2:
                continue
            out.append(Story(chain, row["target_text"].strip(), edges, qedge))
    return out

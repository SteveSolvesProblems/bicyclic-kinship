"""A sound, set-valued solver for relational composition, derived from opaque symbols.

Give it products `a o b = c` and it derives the coordinates that explain them; give it a
chain of any length and it names every relation still consistent with the derived laws.
Nothing in it knows what a relation means. On CLUTRR, trained on chains of length 2 and 3,
it is 100% sound on all 2929 held-out stories of five independently generated instances and
99.80% top-1 on the lengths it never saw.

    from bicyclic_kinship import Solver

    s = Solver.train([(("father", "father"), "grandfather"), ...])
    s.solve(("mother", "son"))     # what the laws admit  -- sound
    s.best(("mother", "son"))      # what the evidence favours -- a preference
    s.explain(("mother", "son"))   # why
"""
from .cone import Cone, cone_axes, cone_compose, cone_fold
from .derive import (additive_axes, ambiguous_chains, binary_table, table_conflicts,
                     triples_from)
from .projection import n_classes, projection_axis
from .pathrules import PathRules
from .solver import Solver
from .system import AxisSystem, collisions

__version__ = "0.1.0"
__all__ = [
    "Solver", "AxisSystem", "PathRules", "Cone",
    "cone_compose", "cone_fold", "cone_axes",
    "additive_axes", "projection_axis", "n_classes", "collisions",
    "binary_table", "table_conflicts", "triples_from", "ambiguous_chains",
]

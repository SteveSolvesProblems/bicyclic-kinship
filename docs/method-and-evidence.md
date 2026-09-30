

Technical companion · bicyclic-kinship

# Deriving Composition Laws, and Trying to Break Them

By **Steve Ball** · September 2026

A model that derives its own rules is only as good as the effort spent trying to prove those rules wrong. This is the record of that effort.

The technical companion to [*Who Is Your Daughter's Grandfather?*](clutrr-explained.html), which explains the model in plain language. This page gives the mathematics, the measurements, and the process behind them: what was tried and dropped, how the model was stress-tested, which assumptions were audited, which claims were corrected, and how the published literature was used to place the work.

**How this was built.** Built with Claude Code, using Claude Opus models. The research direction and the claims are the author's. The explainer carries the full development record: commits, tests, revisions and corrections.

Contents

1.  [Why this work exists](#why)
2.  [The claims, and where each is checked](#ledger)
3.  [The model, stated precisely](#model)
4.  [Four families, each solved exactly](#families)
5.  [Algorithms](#algorithms)
6.  [How the approach was chosen](#path)
7.  [Guarding against my own knowledge](#guard)
8.  [How measurements are made](#discipline)
9.  [Results on CLUTRR](#results)
10. [What each part is worth](#ablation)
11. [Trying to break it](#break)
12. [A worked investigation: the in-law gap](#inlaw)
13. [Placing the work](#context)
14. [What is new, and what isn't](#novelty)
15. [Corrections, and what caught each](#corrections)
16. [Threats to validity](#threats)
17. [What would test it further](#next)
18. [Reproducing everything](#repro)

## Why this work exists

A system that reasons over relations (chaining *A is B's mother* and *B is C's brother* into a new fact) needs rules for how relations combine. There are three ways to get them.

- **Write them by hand.** Logic programs do this, and they are exact: a system that is given the family rules as a logic program handles CLUTRR's text setting well, and doesn't bother with its extrapolation task, because hand-written rules extrapolate by construction ([Yang, Ishay & Lee, 2023](https://arxiv.org/abs/2307.07696)). Anthropology did the same for kinship forty years ago ([Read, 1984](https://doi.org/10.1086/203160)). The cost is that someone who already knows the domain has to supply the knowledge, and the system can't tell you when that knowledge is wrong.
- **Learn them statistically.** Neural and neuro-symbolic systems (R5, NCRL, EpiGNN) learn rules from examples with no domain knowledge. The cost is that nothing states in advance what they will do on an input unlike their training data, their errors are found one input at a time, and their answers come with a confidence score rather than a guarantee.
- **Derive them.** Treat the observed facts as constraints and solve exactly for every rule of a given shape that fits them. If this works, it would combine the strengths of the other two: no domain knowledge, like learning, and rules that are exact, readable and checkable, like writing them by hand.

This project tests the third route, and it exists because the third route gives something the other two don't: answers that can be *audited on inputs nobody has seen*. A derived rule is a claim about every chain, including ones never observed, so it can be read, tested against new data, and falsified by a single counterexample. The explanation of an answer is the computation that produced it, not a story told afterwards. And when the rules don't settle a question, the model returns every answer they allow, instead of picking one and sounding sure.

CLUTRR ([Sinha et al., 2019](https://arxiv.org/abs/1908.06177)) is the right first test for three reasons. It was built to separate systems that learn rules from systems that fit the training distribution, by testing on longer chains than any in training. Its answers are checkable, with no room for a plausible-sounding story. And it comes with two kinds of ground truth: published results from the best learned systems, and an algebra that anthropologists built by hand, which a derivation should recover if it has found something real.

The thesis the evidence here supports is narrow: **on a problem that has a composition law, the law can be derived rather than learned or written, and the derived model reaches the level of the best trained systems while reporting what it does not know.** What it does not settle is how many problems have such a law. That is an empirical question, and the method is cheap enough to ask it one domain at a time. This page is the evidence for the narrow claim, and the record of how it was checked.

## The claims, and where each is checked

Every claim this work makes is listed here, with the measurement behind it and the command or test that reproduces it. Everything else on the page is support for one of these rows, or an attempt to break one.

| Claim | Evidence | Reproduced by |
|----|----|----|
| The answer set contains the true answer on every held-out CLUTRR story | 100% sound on 2,929 stories across five datasets, and on the sixth | `benchmark`; `test_clutrr_soundness_is_total_on_every_instance` |
| It reaches the level of the best published systems on chains longer than any in training | 99.80% top-1 at lengths 4–10; 99.39% at 5–10 on the second split | `benchmark`; [results](#results) |
| It sits at the ceiling any chain-only method can reach | Exactly at it on four of six datasets; 0.17% and 1.4% below on the other two | `benchmark`; `test_clutrr_sits_at_the_ceiling` |
| No domain knowledge is in the code | The same code runs on six synthetic algebras, deriving the right laws or, where none exist, none | the first half of `tests/` |
| It recovers the kinship algebra anthropology built by hand | The derived coordinates are Read's (1984) ascent and descent | `test_clutrr_the_cone_coordinates_are_the_anthropological_ones` |
| Chain length does not degrade it | 100% at 1,000 steps; exact addresses at 10,000 | `worlds`; three depth tests |
| False input facts are flagged, not answered wrongly | 0.0% silently wrong with up to ten false facts per story | `noise`; two noise tests |

Two things are deliberately not claimed. The rules are *audited* against the data, not proved true of the world ([section 11](#break) shows exactly where that difference bites). And the model composes relation symbols; it does not read text.

The four commands are `python -m bicyclic_kinship.benchmark`, `.noise`, `.worlds` and `.stress`. Nothing is averaged over runs, because nothing is random: the derivation is deterministic, and every seeded experiment fixes its seeds.

## The model, stated precisely

Let `Σ` be a finite set of opaque symbols (relation words, with no meaning attached) and let composition be a partial operation on them. The evidence is a set of **triples** `T ⊆ Σ × Σ × Σ`, where `(a, b, c) ∈ T` asserts `a ∘ b = c`. The task: given a chain `w = r₁r₂…r_k`, name the composite `r₁ ∘ … ∘ r_k`. On CLUTRR, `|Σ| = 20` and `|T| = 62` of a possible 400, read off 9,074 training chains of length two and three.

Two facts shape everything that follows. Composition is *partial*: most pairs are never observed, and some composites have no name in `Σ`. And the answer is not always a function of the chain, because two chains of the same words can denote different people. Any method that returns one name is therefore sometimes wrong by construction.

The usual approach is to choose a representation (a vector space, a network) and fit its parameters until `T` is reproduced. This model takes the other approach: **fix a small menu of law shapes, and return every law of each shape that is consistent with every observation.** The model is that solution set. There is nothing to initialise and nothing to tune. The only design decision is the menu, which is stated rather than hidden.

**Definition 1 — a law**

A **law** is a map `φ : Σ → V` into a set with an associative operation `⊙`, such that `φ(a) ⊙ φ(b) = φ(c)` for every `(a, b, c) ∈ T`. It is **valid** if the identity holds for every true product, not just the observed ones.

Associativity is what makes a law usable on a chain. The fold `φ̂(w) = φ(r₁) ⊙ … ⊙ φ(r_k)` doesn't depend on bracketing, so a law derived from pairs applies at every length with no extension. That is the whole mechanism behind length extrapolation. There is no separate story for it.

**Definition 2 — address and admitted set**

Given laws `φ₁ … φ_m`, the **address** of a chain is `α(w) = (φ̂₁(w), …, φ̂_m(w))`, and the **admitted set** is `A(w) = { s ∈ Σ : φᵢ(s) = φ̂ᵢ(w) for all i }`.

**Proposition 1 — soundness**

If every law is valid and composition is single-valued, the true composite of `w`, when it has a name, is in `A(w)`.

By induction on length. If the prefix `r₁…r_{k−1}` has true composite `c′` with `φ̂ = φ(c′)`, validity applied to the true product `c′ ∘ r_k = c*` gives `φ(c′) ⊙ φ(r_k) = φ(c*)`, and the left side is `φ̂(w)`.

The proposition carries two preconditions, and both are treated as things to check rather than assume. *Validity* is a property of the world, which can only be audited against the data in hand ([section 5](#audit)). *Single-valuedness* matters because where a pair has two true composites, the induction step has nothing to apply validity to. It is checked: `table_conflicts` reports zero conflicting products on all five CLUTRR instances. [Section 11](#audit-margin) measures what happens in a world where it fails.

**Proposition 2 — monotonicity**

Removing a law can only widen `A(w)`, since the set is defined by a conjunction of equality tests. Every subset of a sound derivation is sound, so an ablation can cost precision and never soundness.

**Definition 3 and Proposition 3 — preference**

A **preference** is any map with `π(A, w) ⊆ A`. However preferences are composed or ordered, the result is a subset of `A(w)`, so a preference can cost precision and never soundness.

This is why the model reports two things. `solve(w)` returns `A(w)`, what the laws admit. `best(w)` applies preferences and returns one name, labelled as a preference. Keeping them apart was itself a finding: the only genuine errors this project ever had came from merging them, reading a single observed product such as `(daughter, grandmother) = mother` as a function when longer chains show the same address answering `mother-in-law` too.

## Four families, each solved exactly

Each family is a constraint system with a closed-form or exhaustive solution: the derivation returns every member, or reports that there are none. That property is what separates this from searching for structure. An earlier search for symmetries of the same 62 products returned fifty, none of them the one wanted, because a partial table admits any number of accidental symmetries in its weakly constrained corners.

### The additive family: a null space

Take `V = ℚ` with `+`. Build an integer matrix `M` with one row per triple (`+1` for `a`, `+1` for `b`, `−1` for `c`). The additive laws are exactly `ker M`, computed by exact rational elimination. Its dimension, `|Σ| − rank M`, is a property of the data, not a setting. It comes out as 1 for kinship (generation: parents +1, children −1), 1 for filesystem paths, 2 for a grid, and 0 for an algebra with no additive structure.

A law that wraps is the sharpest boundary here. `ℤ/5` is purely additive, but its rational kernel is zero, because a rational null space sees only torsion-free invariants. The model abstains there and stays sound, and a test pins that behaviour.

### The cone: the bicyclic monoid

Kinship composition *cancels*: a mother's son is a brother, not a "mother-son". The first question was whether any invertible representation (phases, rotations, permutations, invertible matrices) could express that. None can:

**Proposition 4 — no group representation**

If an algebra has `x ∘ x = x` and some `y ∘ x ≠ y`, it has no injective homomorphism into a group.

In a group `x² = x` forces `x = e`, so `y ∘ x = y` for every `y`.

Both premises are measured in the corpus: `brother ∘ brother = brother` (67 occurrences, no exceptions) and `father ∘ brother = uncle` (66, none). Since `x ∘ x = x` is transitivity, the argument rules out invertible representations for any relational algebra with a transitive relation in it. The knowledge-graph literature had already reached this conclusion ([section 13](#context)), and this page claims only the check against observed products.

**Definition 4 — the cone**

`V = ℕ²`, with `(u₁, d₁) ⊙ (u₂, d₂) = (u₁ + (u₂ − d₁)⁺, d₂ + (d₁ − u₂)⁺)`. A symbol carries how far a path rises and how far it falls. Composition translates, and folds back onto the boundary of the quadrant.

**Proposition 5 — this is the bicyclic monoid**

Map `(u, d) ↦ pᵘqᵈ` into `B = ⟨p, q | qp = 1⟩`. Every element has the normal form `pᵘqᵈ`, and `qᵈpᵘ = p^{(u−d)⁺} q^{(d−u)⁺}` gives exactly `⊙`. So `⊙` is associative, with identity `(0, 0)`. `B` embeds in no group: `qp = 1` would make `pq = 1`, but `pq` is in normal form and is not `1`.

**Proposition 6 — the linear shadow**

`δ(u, d) = d − u` is a homomorphism onto `(ℤ, +)`. So every cone law's shadow is an additive law, and lies in the kernel already computed.

Proposition 6 makes the search finite: rather than search `ℕ^{2|Σ|}`, the derivation enumerates small integer members of `ker M` and solves for `u` alone. It also explains a puzzle from early in the project. Generation, the first law found, was the visible one-dimensional shadow of a two-dimensional structure. That is why it existed, and why it was not enough.

**Proposition 7 — gauge**

Translating every symbol by `(k, k)` commutes with `⊙`, so assignments are reported up to that translation, pushed down until some symbol touches the boundary.

On CLUTRR the derivation returns four assignments up to gauge. One is the anthropological one (father at `(1, 0)`). On every depth walk measured, the other three place the endpoint at the anthropological address shifted by exactly `(1, 1)`. They disagree only about whether a chain can cancel all the way back to the identity, which no CLUTRR product settles. All four are kept, and a name is admitted only where all four agree.

### Classes: the last step decides, or the first

Take `V = Σ/∼` with right projection, `(x, y) ↦ y`. The law `cls(a ∘ b) = cls(b)` says some property is carried by the last step alone. Gender is the obvious example: whoever a chain passes through, `… ∘ mother` names a woman.

**Proposition 8 — the finest valid partition**

Let `∼` be generated by `c ∼ b` for each `(a, b, c) ∈ T`, closed by union-find. Then `∼` satisfies the law, and every partition that satisfies it is a coarsening of `∼`.

On CLUTRR the partition has six blocks: `{husband, son-in-law}`, `{wife, daughter-in-law}`, `{father, father-in-law, grandfather}`, `{mother, mother-in-law, grandmother}`, `{brother, son, grandson, nephew, uncle}` and `{sister, daughter, granddaughter, niece, aunt}`. A hand-written male/female split is its two-block coarsening, so the derived law is strictly stronger than the domain knowledge it replaces. The fourth family is the mirror image, left projection (the first step decides). CLUTRR supports no such law and the derivation drops it. On a real family tree it finds one with two blocks ([section 11](#tree)), which is the evidence that the family earns its place on the menu.

## Algorithms

### Sparse incremental elimination

A row built from a triple has at most three non-zero entries however large `Σ` is, and there are far more rows than the rank can ever be. Dense elimination spends nearly all its time on zeros, so rows are reduced as they arrive and discarded when they collapse:

    pivots : column → fully reduced sparse row
    for each row r:
        while r touches any pivot column c:        # every one, not only the leading one
            r ← r − r[c] · pivots[c]
        if r = 0: continue                         # most rows end here
        c ← min(support(r));  r ← r / r[c]
        for each pivot row p with p[c] ≠ 0:        # keep the basis reduced
            p ← p − p[c] · r
        pivots[c] ← r

The arithmetic is exact `Fraction` throughout, so this is the same reduced echelon form, not an approximation of it. The first version eliminated only a row's *leading* pivot. It was fast and quietly wrong, and dropped CLUTRR top-1 from 99.4% to 26.1%. The dense implementation is kept in the source for exactly that reason, and a test holds the two to exact equality on six worlds. A fast path needs something that can disagree with it.

### Cone search, and how deep to look

For each candidate shadow `δ ∈ ker M`, assign `u` symbol by symbol, most-constrained first. Assigning two operands forces their product, so the search is mostly propagation and backtracking stays shallow. Within its depth bound the search is exhaustive, so the result is the complete solution set up to gauge.

The bound is read from the data, not fixed, and deciding that was a correction. An early write-up said the evidence left "a family of four" assignments. It doesn't: four was a truncation constant in the search. Search deeper and CLUTRR admits one assignment at depth 2, four at 3, ten at 4, thirty-five at 6 and eighty-four at 8. So the search escalates from depth 1 and stops at the shallowest depth that yields anything: 2 for CLUTRR, 4 for a blood family tree, `k` for a path world capped at `k`.

It then consults one depth further, which gives CLUTRR its four assignments. The reason is specific. The single shallowest law says `husband ∘ wife = wife`, which is false: the truth is *self*, for which CLUTRR has no word. Neither spouse product is ever observed, so no audit can retire the false law. One depth further, the four admitted assignments disagree on exactly that cancellation, and the answer becomes "none of these twenty". Held-out scores are identical either way. The margin is bought entirely off the benchmark, and a test pins it.

### The audit

Laws are derived from products of length two, so validity beyond them is not guaranteed. Each family is checked against every training chain *of every length*, and retired if a single one contradicts it (an opt-in tolerance relaxes this for noisy labels; [section 11](#labels)).

**Proposition 9 — necessary, not sufficient**

A valid law passes the audit. A law that passes the audit can still be invalid, and looks exactly as exact as a valid one.

This is not a technicality. Section 10 contains a law that passes its audit on the entire benchmark and is false about families.

### Preference

By Proposition 3, anything that only narrows is safe, so the preference layer is allowed to be heuristic. It has two sources. *Path rules* are the smallest trigger sets whose effect on a derived coordinate separates training chains that share an address. They are found as a hitting-set check with support and coverage floors, and both floors were added after failures measured during development: without them, the rules memorised (189 held-out errors) or were pure on training and wrong held out (25). *Frequency at the address* counts how often training saw each name at the chain's address, over chains of every length. Ties between equally good triggers break in sorted order; without that, one experiment's numbers varied with Python's string-hash seed (correction 7 in [section 15](#corrections)).

## How the approach was chosen

The model above is what was left after several other approaches were tried and dropped. Most of the process was removal, and the order matters: each step was forced by a measurement on the previous one.

**Before this repository**

- **1. Invertible representations** \[refuted\] — Phases, rotations, permutations: composition as a reversible movement. Three lines of arithmetic (Proposition 4) ruled out the whole family before any code was written.
- **2. Learned factorisation** \[dropped\] — Fitted all 62 products and contained nothing: gender alignment +0.03 against ±0.40 for random pairs; 91.9% interpolating, 35.6% extrapolating.
- **3. Searching for symmetries** \[dropped\] — Fifty symmetries of the partial table, none of them gender. Verifying a symmetry is easy; finding the real one among coincidences is not.
- **4. Transcribed coordinates** \[deleted\] — Generation numbers and gender sets typed into the source. It worked, and it was a transcript of the author. Deleted by rule.

**The model**

- **5. Solve for laws, don't fit them** \[kept\] — A null space and a union-find: exactly one additive law, six classes, and a derived partition finer than the hand-written one.
- **6. The cone** \[kept\] — Cancellation, as the bicyclic monoid. Single-name answers rose from 54.7% to 88.1% at unchanged soundness.
- **7. Geometry, and three extra layers** \[demoted\] — A torus version agreed on every chain and added nothing, so it was demoted. Memory, closure and propagation changed no answer, so they were removed.

**Testing it**

- **8. Leave the benchmark** \[kept\] — A real family tree broke a law CLUTRR cannot falsify. Soundness became an audited property, stated as one.
- **9. The prior-art search** \[reframed\] — Static analysis, not verification. Proposition 4 already known. A claim with no source removed.
- **10. Sixth dataset, depth, two extensions** \[added\] — The second extrapolation split, walks of 10,000 steps, route memory and word senses, each passing the synthetic controls first.
- **11. A script for every table** \[added\] — Writing this page found tables no script reproduced. Re-measured, two changed and one claim was false.

A note on provenance. The measurements in the first four steps were made in two earlier research repositories this work grew from, and they cannot be reproduced from this one. They are offered as history, to show how the design was arrived at, not as evidence for any claim in [the ledger](#ledger). Everything from step five on is reproducible here.

Three habits came out of that sequence and shaped everything after it:

- **Make the premise falsifiable before making it work.** The invertible-representation idea was killed by three lines of arithmetic (Proposition 4) before any code was written. Weeks of tuning would have produced a mediocre number and no understanding of why.
- **Fitting is not evidence.** A factorisation of the 62 products reproduced every one and contained nothing. Male/female differences in the learned space had alignment +0.03, against ±0.40 for random pairings. It interpolated a hidden product at 91.9% and extrapolated at 35.6%. Reproducing training data is what optimisation does, so the useful test is whether the fit *contains* the structure, probed directly.
- **A favourite idea can be demoted without being deleted.** A geometric version (the laws as lanes on a torus) agreed with the algebra on all 1,146 held-out chains, and did nothing the algebra doesn't. It was demoted to a corroborating implementation, with a stated price for its return: a measurement where it does something the algebra can't. It is not in this repository.

## Guarding against my own knowledge

The easiest way for a "derived" model to cheat is for its author to know the answer. I know what `father` means, and every shortcut I take with that knowledge makes the result less meaningful. These are the guards, and the incidents that made each one necessary.

- **The derivation ships as the derivation.** At one point generation numbers and gender sets, established in a throwaway script, were typed into the program as a dictionary. The program worked, and every measurement after that was reported against code that could not have produced those numbers on any other input. The file was deleted, and the rule since then is that a literal the code could compute does not ship.
- **Variable names can carry knowledge.** A gender result was withdrawn when it turned out to depend on word pairs being listed male-first. The pairing had no such order in the data; the order came from how I typed it.
- **The code is scanned for domain words.** The derivation and solving modules contain no kinship words outside comments and docstrings. That is checked by scanning the source, not asserted.
- **Synthetic controls come first.** The first half of the test suite runs the same code on algebras with no people in them: a stack of pushes and pops, an additive ladder, left- and right-zero semigroups, a structureless magma and `ℤ/5`. Each one checks that the right family is found, or that none is. If a mechanism only works on kinship, these fail.
- **Each extension passes the same controls.** Route memory and word senses ([section 12](#inlaw)) had to derive *nothing* on worlds without the structure they look for before any kinship result counted.
- **Lanes are measured stripped.** A lane's correctness is measured on its own, not through the assembled model, because the other lanes distort the reading in both directions. At the shallowest depth the cone alone admits both spouses, and the class lane then narrows that to the single, wrong name `wife`. It does not fix the error; it makes the error confident.

## How measurements are made

- **A number no script reproduces is a claim, not a measurement.** For a while early on, every end-to-end number came from a shell script that was thrown away, while the modules it measured were imported by nothing. Since then every reported figure is printed by one of four scripts or asserted by a test. Writing this page tested that rule again: several tables in the older writeups had no script behind them. They were re-measured by a new one (`stress`), two of them changed, and one claim turned out to be false (corrections 9 and 10).
- **Deterministic, and checked to be.** All reported numbers were re-run under eight different Python hash seeds. That check found one experiment whose numbers moved, traced it to an unordered tie-break, and fixed it.
- **Held-out data is touched once.** Candidate features are selected on a validation split carved out of *training*, and only then scored held out. [Section 11](#bait) shows a feature this protocol correctly rejected.
- **A score comes with its ceiling.** Because the corpus answers some chains two ways, every score is reported next to the best any function of the chain could reach. Anyone can compute that ceiling from the test set alone, without trusting the model.
- **Comparisons are checked for like-for-like.** Before comparing with published systems, I read their papers for the input format ([section 13](#context)). One early claim, that the comparison was meaningless, did not survive that reading.

## Results on CLUTRR

CLUTRR ([Sinha et al., 2019](https://arxiv.org/abs/1908.06177)) has a text setting and a symbolic one. The best systems are measured symbolically, on the relation graph with only the required facts. This model uses the same input. The public release has six datasets.

All six datasets. The model is trained once on `gen_train23`, except for the second extrapolation split, which is trained on its own data as its published protocol requires.

| Dataset                             | Stories | Sound | Top-1   | Ceiling | Mean set |
|-------------------------------------|---------|-------|---------|---------|----------|
| `gen_train23`, all lengths          | 1,146   | 100%  | 99.39%  | 99.56%  | 1.12     |
| `gen_train23`, unseen lengths 4–10  | 1,003   | 100%  | 99.80%  | —       | —        |
| `gen_train234`, all lengths         | 1,048   | 100%  | 98.09%  | 99.52%  | 1.11     |
| `gen_train234`, unseen lengths 5–10 | 826     | 100%  | 99.39%  | 100%    | —        |
| `rob_train_clean`                   | 447     | 100%  | 100.00% | 100.00% | 1.65     |
| `rob_train_disc`                    | 445     | 100%  | 94.83%  | 94.83%  | 1.65     |
| `rob_train_irr`                     | 444     | 100%  | 93.69%  | 93.69%  | 1.91     |
| `rob_train_sup`                     | 447     | 100%  | 93.74%  | 93.74%  | 1.78     |

The derived laws are identical on all five `gen_train23`-style instances, down to the cone assignment: retraining on each one's own data changes nothing. The four `rob_*` datasets test distracting sentences in the prose. For a model that never reads prose, they are four more independent samples of chains of length two and three, so they test consistency, not extrapolation.

### Next to published systems

Top-1 by chain length, trained on 2–3 steps. Published numbers are means over runs, from R5 ([Lu et al., 2022](https://arxiv.org/abs/2205.06454)), Table 2.

| Steps                       | 4    | 5    | 6   | 7   | 8   | 9   | 10  |
|-----------------------------|------|------|-----|-----|-----|-----|-----|
| This model                  | 99.5 | 99.4 | 100 | 100 | 100 | 100 | 100 |
| R5 (ICLR 2022)              | 98   | 99   | 98  | 96  | 97  | 98  | 97  |
| CTP<sub>A</sub> (ICML 2020) | 99   | 99   | 99  | 96  | 94  | 89  | 90  |
| GAT                         | 91   | 76   | 54  | 56  | 54  | 55  | 45  |

Trained on 2–4 steps. Published numbers from EpiGNN ([Khalid & Schockaert, 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/6590cb829f5ffef50050f3e5845fbb4c-Paper-Conference.pdf)), Table 1.

| Steps              | 5    | 6   | 7    | 8   | 9   | 10  |
|--------------------|------|-----|------|-----|-----|-----|
| This model         | 98.4 | 100 | 98.7 | 100 | 100 | 100 |
| NCRL (ICLR 2023)   | 100  | 99  | 98   | 98  | 98  | 97  |
| R5 (ICLR 2022)     | 99   | 99  | 99   | 100 | 99  | 98  |
| EpiGNN (ICLR 2025) | 99   | 99  | 99   | 99  | 96  | 98  |

The honest reading is **the same level as the best published systems, not better**. Three reasons. The top systems sit within their own run-to-run spread (±1–5%) of 100%. The published numbers use CLUTRR's original data release, and these runs use the HuggingFace release of the same splits: generated the same way, but not identical test sets. And a difference of one or two stories per length is not a result. What the comparison does support is narrower and still interesting: a model with no weights, reading opaque symbols, reaches the level of trained neuro-symbolic rule learners, and reports which answers it is unsure of.

### Where the misses are

Every miss was examined, not just counted. On `gen_train23`, 7 of 1,146 stories are missed. Five are the corpus contradicting itself: `wife · son · grandmother` appears in the test set with both answers, and no function of the chain can get both. The other two reduce in the algebra to the same shape (a spouse's child is your child, whose grandmother is your mother or your spouse's). On `gen_train234`, every miss at every length is the same kind: the answer set is correctly `{mother, mother-in-law}` or `{father, father-in-law}`, and the preference picks the blood term when the label is the in-law one.

One result there runs against intuition. On the `gen_train234` unseen lengths, the model trained on its own 2–4-step data scores 99.39%, and the model trained on the shorter `gen_train23` data scores 100%. The laws are identical, so the difference is entirely in the preference: the extra four-step training chains shift the frequency counts at the two ambiguous addresses. More data made a heuristic slightly worse, and the soundness guarantee was untouched either way. It is reported as observed.

## What each part is worth

Proposition 2 predicts that every ablation stays 100% sound, so what ablation measures is precision.

On `gen_train23`, held-out lengths 2–10. From `stress`.

| Laws                   | Sound | Top-1 | One name | Mean set |
|------------------------|-------|-------|----------|----------|
| All families           | 100%  | 99.4% | 88.1%    | 1.12     |
| Without the cone       | 100%  | 72.3% | 54.7%    | 1.45     |
| Without the classes    | 100%  | 87.3% | 0.0%     | 2.39     |
| Without the additive   | 100%  | 98.7% | 88.1%    | 1.12     |
| The cone alone         | 100%  | 87.3% | 0.0%     | 2.39     |
| Without the path rules | 100%  | 98.8% | 88.1%    | 1.12     |

The cone is worth 27 points here, and 10.6 across all five instances (97.06% against 86.45%). The classes carry every male/female distinction, so without them nothing is ever pinned to one name. The additive family is nearly redundant once the cone exists, as Proposition 6 predicts, since it is the cone's shadow. The whole preference stack is worth 0.6 points.

Three further layers were measured on all five instances and removed, because they changed *no answer* once the cone existed: a memory of witnessed transitions, closure over observed composites, and unit propagation of the observed table. More expressive law families were tried and likewise changed nothing on CLUTRR. A layer that has stopped paying for itself is a finding about the model, and it doesn't ship.

## Trying to break it

The headline numbers say the model works on the benchmark it was built for. The questions below are the ones a sceptical reviewer would ask next, and each was answered by a measurement designed to make the model fail. Some of them did.

### "The training data is basically the answer"

Sixty-two products out of four hundred sounds like a lot of help. So take products away, choosing five seeded random subsets at each size, and see what survives.

*\[figure — see the HTML version\]*

**Soundness needs enough evidence to kill false laws.** Each point is the mean of five seeded random subsets of the 62 training products. With 15, the derived laws are exact, confident and wrong on nine chains in ten. Soundness is not a property of the method alone.

| Products kept, of 62 | 15   | 24    | 31    | 37    | 46    | 55    | 62    |
|----------------------|------|-------|-------|-------|-------|-------|-------|
| Sound                | 9.8% | 52.7% | 74.5% | 83.5% | 96.6% | 100%  | 100%  |
| Top-1                | 9.8% | 52.2% | 72.2% | 82.5% | 95.9% | 99.3% | 99.4% |
| Chains it can read   | 767  | 946   | 967   | 1,051 | 1,144 | 1,146 | 1,146 |

This is the most uncomfortable result on the page. With a quarter of the products the derived laws are **confidently wrong**, and nothing warns you, because a law fitted to too little evidence looks exactly as exact as a true one. Soundness is not a property of the method alone. It is a property of the method *plus enough data to falsify the laws that are false*. The derivation cannot tell you which regime you are in.

### "Exact derivation must be brittle to noisy labels"

It is. Replace a share of training answers with a random word:

*\[figure — see the HTML version\]*

**Exact laws are brittle; the tolerance is not.** One corrupted answer in two hundred retires every exact law. From one in a hundred, corruption also deletes words, and the share of held-out chains the exact model can read at all (dotted) falls to zero by 10%. The opt-in tolerance holds top-1 at 98.8% through 10% and gives up at 20%.

| Corrupted                 | 0%    | 0.5%  | 1%    | 2%    | 5%    | 10%   | 20%  |
|---------------------------|-------|-------|-------|-------|-------|-------|------|
| Exact: words left         | 20    | 20    | 17    | 15    | 5     | 0     | 0    |
| Exact: chains it can read | 100%  | 100%  | 84.8% | 71.3% | 0.3%  | 0%    | 0%   |
| Exact: sound, on those    | 100%  | 100%  | 100%  | 100%  | 100%  | —     | —    |
| Exact: top-1              | 99.4% | 6.5%  | 7.7%  | 5.2%  | 0.1%  | 0%    | 0%   |
| Tolerant: top-1           | 99.4% | 98.8% | 98.8% | 98.8% | 98.8% | 98.8% | 6.5% |
| Tolerant: sound           | 100%  | 100%  | 100%  | 100%  | 100%  | 100%  | 100% |

One corrupted answer in two hundred retires every law, because a single contradicting chain is enough and there are over nine thousand of them. There is a second effect, found while writing this page. A product whose witnesses disagree is dropped, and a word with no surviving product leaves the vocabulary. At 1% corruption three words are gone; at 10%, all of them. The older writeup said soundness "stays at 100% throughout". That was measured only on the chains the model could still read, which by 5% is three stories. On those it is still sound; on the rest it can't answer at all. The table now shows both, and a test pins the vocabulary loss.

The opt-in `tolerance=0.1` settles each product by majority and keeps a law while the share of chains contradicting it stays under 10%. On clean data it changes nothing, to every decimal place. It holds through 10% corruption and gives up at 20%, by abstaining. The default stays at zero, because a tolerance is exactly what lets a false law survive its audit.

### "What if the input facts are wrong?"

CLUTRR's own noise lives in the prose, so it was rebuilt on the symbolic side, in a stricter form: extra edges are added to each story's graph, and the model must find every route between the two people itself, compose each, and intersect the answer sets. Intersection is the sound combination, since extra true facts can only narrow.

| Extra facts per story        | 0     | 1     | 2     | 5     | 10    |
|------------------------------|-------|-------|-------|-------|-------|
| True but irrelevant: sound   | 99.8% | 99.8% | 99.8% | 99.8% | 99.8% |
| False: answered and sound    | 99.8% | 68.2% | 45.6% | 13.1% | 2.0%  |
| False: contradiction flagged | 0.2%  | 31.8% | 54.4% | 86.9% | 98.0% |
| False: silently wrong        | 0.0%  | 0.0%  | 0.0%  | 0.0%  | 0.0%  |

Irrelevant facts change nothing, exactly, and the test asserts equality, not a tolerance. False facts can destroy an answer, but two routes that disagree intersect to the empty set, which is a detected contradiction. (The 0.2% baseline is two stories whose graphs revisit a person, which creates a short cut whose composite has no word in the vocabulary.) The rate of confidently wrong answers stays at zero. That was not tuned in; it follows from soundness plus set-valued answers.

### "The benchmark is one generator"

So the model was taken somewhere CLUTRR can't follow: a family tree built generation by generation and walked at random. The true relation comes from graph distance (nearest common ancestor, steps up, steps down), never from composing the chain. Walks never meet the same person twice, which is the condition under which the words determine the answer at all.

| Chain length                        | 2     | 3     | 5     | 8     |
|-------------------------------------|-------|-------|-------|-------|
| Walks                               | 1,211 | 981   | 393   | 10    |
| Laws from CLUTRR: sound             | 87.0% | 78.9% | 81.7% | 90.0% |
| Laws from the tree: sound and top-1 | 100%  | 100%  | 100%  | 100%  |

This is where "audited, not proved" gets a number. In the entire CLUTRR corpus, train and test, every chain ending in `husband` answers `son-in-law`, because the only such chain that ever occurs is `daughter · husband`. So the class law files `husband` with `son-in-law`, passes its audit on every chain the benchmark contains, and is false about families: your mother's husband is your father, and the CLUTRR-derived laws admit nothing there. Trained on the tree's own 48 products, the six classes coarsen to four, a left-projection law appears with two classes, and every answer on walks through a different tree is right.

Is that one lucky tree? Five trees, 500 two-step walks each, from `stress`.

| Tree seed                 | 5     | 42    | 99    | 1234  | 7777  |
|---------------------------|-------|-------|-------|-------|-------|
| People                    | 2,046 | 1,132 | 2,596 | 2,440 | 1,216 |
| Laws from CLUTRR: sound   | 88.2% | 87.4% | 85.8% | 85.0% | 85.4% |
| Own laws: sound and top-1 | 100%  | 100%  | 100%  | 100%  | 100%  |
| Classes derived           | 4     | 4     | 4     | 4     | 4     |

The lesson is general, and it is the most important one in this work: **a benchmark's coverage decides which laws can be derived from it, and no audit against that benchmark can see past it.** The only defences are cheap ones: take the model somewhere its benchmark can't follow, and remove data on purpose to watch where it breaks.

### "How deep can it really go?"

Real family trees run out of new people at around twenty steps, so a deeper test needs a family that grows as it is walked: parents are created the first time anyone asks for one, and a new child can always be born. The truth is still read from the family, never computed by composition.

| Chain length | Chains | Sound | Top-1 | Time per answer |
|--------------|--------|-------|-------|-----------------|
| 10           | 300    | 100%  | 100%  | 0.06 ms         |
| 100          | 300    | 100%  | 100%  | 0.25 ms         |
| 1,000        | 100    | 100%  | 100%  | 2.3 ms          |

Those are laws derived from CLUTRR chains of length two and three, never retrained, on walks restricted to relatives CLUTRR has words for. Unrestricted walks of 100, 1,000 and 10,000 steps end on people no word names, such as `(2, 10072)`: two generations up and ten thousand and seventy-two down. The model correctly answers with no name, and the address underneath can be checked directly. On every walk tested it equals the true (ascent, descent) exactly.

### "It only works because kinship is bicyclic"

That is true, and the useful question is what happens elsewhere. The same code, unchanged, on worlds with no people in them:

| World | Symbols | Products | Derived | Result |
|----|----|----|----|----|
| Filesystem paths, one folder name | 16 | 176 | 1 additive, 1 cone | 100% exact at 10,000 steps |
| Filesystem paths, two folder names | 45 | 846 | 1 additive, 1 cone | 100% sound, about 6 names per answer |
| Movement on a grid | 49 | 1,369 | 2 additive | 100% exact at 10,000 steps |
| Kinship, open vocabulary (cousins, removes) | 36 | 160 | 1 additive, 2 cone, 4 + 2 classes | 100% exact to 10 steps |
| S₃, the smallest non-abelian group | 6 | 36 | nothing | 100% sound, 13.0% top-1, all 6 names |
| ℤ/4, a clock | 4 | 16 | nothing | 100% sound, 31.5% top-1, all 4 names |

Each row says something the kinship result can't. Paths are the same monoid (`..` pops, a folder name pushes), and because paths contain a full cancellation back to the identity, they pin the cone assignment *uniquely* where CLUTRR leaves four. That identifies exactly which missing observation CLUTRR would need. Two folder names make a world richer than the law, where `x/y` and `y/x` share coordinates; the model widens its answer instead of guessing. The grid needs two additive coordinates, read from the null space without being told. And the two groups fit no family, so the model derives nothing, admits everything, and says so. That is the intended behaviour at the edge of the menu, and the reason the menu is stated as a prior.

### "Does it scale?"

It did not, at first. A 225-symbol derivation spent all its time in dense rational elimination, and in a second copy of the same elimination inside the cone search. Sparse incremental elimination ([section 5](#algorithms)) fixed both.

One run on a laptop, from `stress --dense`. Absolute times vary between runs; the ratios don't.

| World | Symbols | Products | Whole derivation | Kernel, sparse | Kernel, dense |
|----|----|----|----|----|----|
| Paths, two names | 60 | 1,888 | 0.18 s | 0.04 s | 7.5 s |
| Grid, span 5 | 121 | 8,281 | 1.3 s | 0.22 s | 20 s |
| Grid, span 7 | 225 | 28,561 | 1.4 s | 0.84 s | 194 s |
| Grid, span 10 | 441 | 109,561 | 6.5 s | 3.8 s | — |
| Grid, span 14 | 841 | 398,161 | 27 s | 17 s | — |

At 225 symbols the sparse kernel is over two hundred times faster than the dense one, with identical output. The arithmetic is still exact over the rationals, so this is a constant-factor result, not a complexity one. Answering was never the bottleneck: it is linear in chain length and independent of vocabulary size.

### A feature that won on validation and lost held out

The remaining CLUTRR misses tempt a fix. The model already computes whether a symbol at cone coordinate `(0, 0)` (a spouse, though the code never calls it that) appears in the chain, and "spouse anywhere, so answer the in-law name" looks like it should help at the two ambiguous addresses. It was selected by the protocol in [section 8](#discipline):

| Rule at the ambiguous addresses | Fit (59) | Validation (15) | Held out (136) |
|---------------------------------|----------|-----------------|----------------|
| Always the blood name           | 72.9%    | 80.0%           | 89.7%          |
| Spouse anywhere → in-law name   | 89.8%    | 93.3%           | 87.5%          |
| The shipped preference          | —        | —               | 94.9%          |

The feature beats the baseline on the data it was fitted on and on honest validation, and is worse than doing nothing held out. That is distribution shift, not overfitting, and the cause is measurable. A marriage crossing can be absorbed by a later step down (your wife's daughter is your daughter), and chains of length two and three rarely show it: a spouse appears in 35.1% of these training chains and 22.8% held out. The distinction the rule needs is longer than any training chain of that shape. This is exactly the trap CLUTRR was built to set, and a standard train/validation protocol walks into it. The experiment is kept as a test, so the feature can't be rediscovered and shipped on its validation number.

### The audit and the margin catch different failures

Two defences protect soundness, and it would be easy to assume one covers the other. Both were measured on a case built to break each:

| Case | The audit | The depth margin |
|----|----|----|
| A multi-valued world (a sibling's sibling may be yourself) | Decisive: 34 cone laws retired, soundness 19.2% → 100% | Blind: 19.2% at every margin |
| A product never observed (`husband ∘ wife`) | Blind: nothing contradicts the false law | Decisive: "none of these" instead of a wrong name |

In the multi-valued world, the unaudited cone laws agree with every surviving product and disagree freely where the conflicting ones were dropped. Since a name must satisfy every law, their intersection excludes the *true* answer, and the lane returns nothing on 78% of held-out chains. That is the counterexample to "more laws can only narrow, so more is always safe": narrowing can exclude the truth as easily as a falsehood. The audit refutes laws the data contradicts; the margin covers a law the data is silent about. Neither subsumes the other, and both are tested.

## A worked investigation: the in-law gap

One open problem was followed from first observation to resolution, and my own conclusions about it were wrong twice. It is the clearest example of the process.

1.  **Observation.** CLUTRR never uses an in-law word inside a chain. All four appear only as answers, so nothing in the corpus shows how an in-law relation composes, and whatever the derivation does with one is invented. On a family tree, the invention is wrong.
2.  **First hypothesis: a data gap.** Walk a tree with marriages, where 70 of 182 observed products have an in-law word as an operand. *Confirmed:* soundness returns to 100% at every length.
3.  **Second hypothesis: a structural limit.** Precision did not return. At length five only 54.3% of answers were right, and only 24.6% of in-law answers. `father` and `father-in-law` share an address, and in-law-ness looked like a property of the *path* (did it cross a marriage, and was that crossing later absorbed?) rather than of the endpoints, which is all any family on the menu can express. I wrote that up as a structural limit.
4.  **Test one: is the decay real?** Those numbers came from training on products alone. Trained on walks of length two and three, as CLUTRR itself trains, the preference gets in-law answers 100% right at lengths three and five. *Most of the decay was a training artifact.* What stayed open was knowledge: the laws alone pinned one name on only 9–35% of chains.
5.  **Test two: build the missing kind of law.** If in-law-ness is path state, the model needs a family that carries state. Route memory derives the smallest automaton consistent with every training chain, audited like the other families. It found real structure: on this tree its two states split the vocabulary exactly at "-in-law", a string the code never reads. But it helped far less than predicted, raising single-name answers from 9% to 24% at length two and from 13% to 19% at length three. A more expressive variant overfit and fell to 65% sound; given longer training chains, the audit retired it. On CLUTRR the audit retires route memory entirely, on 52 contradicting chains.
6.  **Test three: a different explanation.** English uses one word for two relations. *Brother-in-law* is both a sister's husband and a spouse's brother, and one word at two addresses collapses the class lane for every in-law term. Word senses let a word that appears only as an answer split into the fewest senses its products require. On the tree it split six in-law words into exactly English's two meanings, found from data alone, and single-name answers went to 100%, still 100% sound, across nine pairs of training and test trees. On CLUTRR nothing splits, and every number is unchanged.

*\[figure — see the HTML version\]*

**The predicted fix helped a little; the unexpected one closed the gap.** Share of chains on a family tree with marriages that the laws alone pin to one name, all at 100% soundness. Route memory was built because the gap looked like path state. Word senses showed it was mostly polysemy.

The gap turned out to be mostly two words hiding in one, not a missing kind of law. The "structural limit" was retracted and the record says so (correction 8). The same senses mechanism, tested on English-style cousin names where "first cousin once removed" means both a parent's cousin and a cousin's child, took single-name answers from 0% to 100% at every length from 2 to 10. Two caveats remain. Tree walks never revisit anyone, so every answer there is fully determined; on CLUTRR, where some chains genuinely have two answers, senses rightly split nothing. And a word that is ambiguous *and* used inside chains is not yet handled.

## Placing the work

After the model worked, I went looking for what it already was. The honest answer is: mostly things with names in other fields. That search changed the claims more than any experiment did, and it is recorded here in full.

### The method: static analysis, not verification

The first framing called this verification. It isn't. Verification checks a given system against a given specification; this infers a specification from observed behaviour, which is the static-analysis tradition. Every write-up now names that lineage.

- **Abstract interpretation** ([Cousot & Cousot, 1977](https://doi.org/10.1145/512950.512973)) is the closest correspondence. Each family is a sound over-approximation, and the families resemble standard abstract domains: the additive family is a linear-equality domain, and `ℤ/5`'s refusal is the known blind spot of a rational domain to congruences. The asymmetry this work leans on (a refusal proves something, an admission does not) is the ordinary soundness argument of static analysis.
- **A precision a reviewer would check.** Intersecting independent lanes is the *direct* product of domains. A *reduced* product exchanges information between components to sharpen each ([Cousot, Cousot & Mauborgne, 2011](https://doi.org/10.1007/978-3-642-19805-2_31)). This model does that only at derivation time, where the cone search consumes the additive kernel, and not at answering time. Claiming the reduced product would be wrong.
- **The audit is specification mining.** Daikon ([Ernst et al., 1999](https://doi.org/10.1145/302405.302467)) proposes candidate invariants and keeps those no execution contradicts. That is this audit exactly, including its status: likely invariants, not proved ones. The false `husband` law is the standard failure of the whole approach.
- **Deriving the domain rather than designing it** is the least standard part, and it is not new either. ATLAS learns abstract domains for program synthesis ([Wang, Anderson, Dillig & McMillan, 2018](https://doi.org/10.1007/978-3-319-96145-3_22)).

### A result that turned out to be known

Proposition 4 was going to be the basis of a short note arguing that group-shaped knowledge-graph embeddings can't represent transitive relations. The prior-art search killed the note. Knowledgebra ([Yang et al., 2022](https://arxiv.org/abs/2204.07328)) makes exactly that argument, that relation composition is a semigroup rather than a group, and proposes a semigroup embedding on that basis. Rot-Pro ([Song, Luo & Huang, 2021](https://arxiv.org/abs/2110.14450)) had hit the same wall in practice and fixed transitivity with an idempotent projection. What remains of Proposition 4 is a two-line argument and a criterion checkable against observed products. The white paper was updated to say so before anything was published.

### The algebra: prior art as ground truth

The bicyclic monoid was first described by Lyapin (1953) and is textbook material (Clifford & Preston, 1961). The algebraic study of kinship goes back to Weil's 1949 appendix for Lévi-Strauss. The closest match is Dwight Read's algebra of American kinship terms ([Read, 1984](https://doi.org/10.1086/203160)), built by hand, which defines a kin-term product with a marker for products that have no term. Its generating equations include *parent of child = self* and *child of parent = sibling*, as quoted in [Read, Fischer & Leaf (2013)](https://doi.org/10.1177/0894439312455914). The first is the bicyclic relation `qp = 1`; the second is the order that does not cancel.

That makes Read's algebra a ground truth, and a better outcome than novelty would have been. A method that recovers, from 62 products over opaque symbols, a structure a field established by hand has found something real, and the prior art is what it can be checked against. An earlier draft said the kinship literature calls this structure a "bicyclic semigroup". No source for that could be found, so the sentence was replaced with Read's own equations (correction 6).

### The comparison: reading the baselines' input format

An early draft said comparing with published systems was meaningless, because they must find the reasoning path in a graph while this model is handed a chain. Reading the papers corrected that. The Edge Transformer paper ([Bergen, O'Donnell & Bahdanau, 2021](https://arxiv.org/abs/2112.00578)) specifies "the noiseless (only the required facts are included) graph-based version", and R5 likewise consumes symbolic triples. With only the required facts present, the graph *is* the chain. The second extrapolation split was added because NCRL and EpiGNN report on it, which let the comparison use the protocol the most recent systems publish. The data-release difference (original versus HuggingFace) was found the same way, and is stated wherever the comparison appears.

Every citation in both write-ups was checked against Crossref or arXiv before publication. Where a claim could not be sourced, it was removed rather than softened.

## What is new, and what isn't

Each claim this work could make is set against the closest prior work I found, with a status. "New" means only that I have not found it; it is the claim most likely to be wrong, and corrections are welcome.

| Claim | Closest prior work | What differs | Status |
|----|----|----|----|
| The kinship algebra is **derived** from observed products over opaque symbols, with no domain knowledge | Read (1984) built it by hand. KAES ([Read & Behrens, 1990](https://escholarship.org/uc/item/43f9d51h)) builds such algebras interactively, with the analyst choosing the generators and equations. Yang, Ishay & Lee (2023) write the family rules as a logic program. Hinton's family-trees network (1986) and linear relational embedding ([Paccanaro & Hinton, 2001](https://doi.org/10.1109/69.917563)) learn kinship from triples, as does FOIL's rule learning ([Quinlan, 1990](https://doi.org/10.1007/BF00117105)) | Nothing is supplied: no generators, no equations, no meanings. The result is exact, and it is the same algebra the hand-built work arrived at | \[new, as far as I found\] |
| Rules that are **neither trained nor written by hand** reach the level of the best published CLUTRR extrapolation results | R5, NCRL and EpiGNN learn their rules; Yang et al. are given theirs | Neither training nor supplied rules. The level is the same as the best systems, not higher | \[new result, as far as I found\] |
| The model is the **complete set of consistent laws**, and a name is admitted only where they all agree | Version spaces ([Mitchell, 1982](https://doi.org/10.1016/0004-3702(82)90040-6)) keep every hypothesis consistent with the data and classify only when all agree. Abstract interpretation intersects sound approximations | The hypothesis spaces are algebraic shapes solved in closed form (a null space, the bicyclic monoid, union-find), so the complete set is computed rather than bounded | \[known idea, new instance\] |
| **Sound, set-valued answers**, reported with set size as the headline | Conformal prediction ([Vovk, Gammerman & Shafer](https://doi.org/10.1007/978-3-031-06649-8)) returns sets with a statistical coverage guarantee. Static analysers report soundness routinely | The guarantee here is conditional on the laws being valid, not on exchangeable data. None of the CLUTRR systems I found reports soundness or set size | \[new to this benchmark, not a new idea\] |
| Deriving the laws **audits the benchmark**: its ceiling, and a law its coverage cannot falsify | Yang et al. found 16 faulty instances in CLUTRR 1.3 by hand, one of them labelled *mother* where *mother-in-law* was right. Path-of-Thoughts ([Zhang et al., 2024](https://arxiv.org/abs/2412.17963)) notes questions with several correct answers and one label | Here the ambiguity is derived rather than found by inspection, measured exactly on all six datasets as a ceiling, and shown to be always a blood/in-law pair. The `husband` coverage artifact is, as far as I found, not reported elsewhere | \[partly known\] |
| Completing a **composition table** from part of it: 62 observed products answer 190 of 400 | Grokking ([Power et al., 2022](https://arxiv.org/abs/2201.02177)) learns binary-operation tables from a fraction of their entries. Semi-automatic methods compute composition tables from a known semantics ([Liu & Li, 2011](https://arxiv.org/abs/1105.4224)) | Exact derivation instead of learning. It also fails where grokking succeeds: modular addition is outside the menu | \[known problem, different method\] |
| No invertible representation can express a transitive relation (Proposition 4) | Knowledgebra (Yang et al., 2022); Rot-Pro (Song, Luo & Huang, 2021) | Only that it is checked against observed products | \[known\] |
| Route memory and word senses | Automaton inference (Gold, 1978; Angluin, 1987); splitting states until a partition is consistent is the Myhill–Nerode construction | Used as audited lanes that must derive nothing on worlds without the structure | \[known techniques, applied\] |

Taken together, the contribution is an **assembly, plus one result**. The assembly: standard algebraic structures, each solved exactly so that the model is the complete set of consistent laws; a strict separation between what those laws admit and what the evidence prefers; and soundness and set size reported where the literature reports only accuracy. A reader who concludes this is a version space over algebraic hypotheses, read as direct-product abstract interpretation with a mined specification, is close enough to right that it should be the starting point rather than something to argue with. The result: that assembly, given nothing but 62 products over symbols it cannot read, recovers the kinship algebra anthropology built by hand and matches trained systems on the benchmark built to test extrapolation. What is not claimed: any new mathematics, any advantage over the best systems beyond their error bars, or generality beyond the worlds tested.

## Corrections, and what caught each

Eleven claims were withdrawn or corrected. They are listed because *how* each was caught is the best evidence of how the work was checked.

| \# | What was claimed | What caught it | Now |
|----|----|----|----|
| 1 | A gender result | Reviewing where the pairing came from: word pairs listed male-first | Withdrawn; gender is whatever the class lane derives |
| 2 | Comparing with published systems is meaningless | Reading the baselines' input specification | Same symbolic input; the comparison stands, "same level" |
| 3 | The data leaves "a family of four" coordinate systems | Reading the search: four was a truncation constant | Depth read from data; one at the least depth |
| 4 | Proposition 4 as a new result | Prior-art search (Knowledgebra, Rot-Pro) | Credited as known |
| 5 | The method framed as verification | Prior-art search | Named as static analysis |
| 6 | Kinship literature calls it a "bicyclic semigroup" | Citation check found no source | Replaced with Read's equations |
| 7 | Published in-law numbers | Re-running under eight hash seeds | Tie-break fixed; numbers deterministic |
| 8 | The in-law gap is a structural limit | The experiments in [section 12](#inlaw) | Narrowed twice; mostly polysemy |
| 9 | Under label noise, soundness "stays at 100% throughout" | Writing a script for a table that had none | True only on words that survive; both now reported |
| 10 | Evidence-removal, tree-seed and cost tables | The same: no script reproduced them | Re-measured by `stress`; with 15 products, sound 9.8%, not 20.7% |
| 11 | Every published CLUTRR system is trained | Writing the novelty comparison: Yang et al. (2023) are given hand-written rules | "Learned or written by hand; none derived" |

None of these changed a headline number. Several changed what the headline numbers mean, and four (1, 3, 9 and 11) had made the work look stronger than it was.

## Threats to validity

- **The menu is a prior.** The coordinates are derived, but which shapes to search (additive, bicyclic, two kinds of class) is a choice I made. It is a small library of standard structures, and should never be framed as "no priors".
- **Soundness is audited, not proved.** On a real family tree the CLUTRR-derived laws are 79–90% sound. The benchmark can't expose that, because it never generates the chain that breaks the law.
- **It needs enough data to kill false laws.** With a quarter of the products it is confidently wrong, and nothing warns you at the time.
- **Exact derivation is brittle to label noise.** The tolerance repairs it, and trades away the guarantee to do so.
- **The generality evidence is mostly worlds I built.** Paths, grids, family trees and the synthetic algebras were written by someone who knew the menu. That is a real limitation. A benchmark with many worlds built by others is the right next test ([section 17](#next)).
- **The comparison uses a different data release.** Same generator, same protocol, different test sets. It supports "the same level", and nothing stronger.
- **The symbols are given.** Someone supplies the vocabulary and the products. Deriving the vocabulary from text or perception is untouched here, and is probably the harder half of the problem.
- **CLUTRR is small.** Twenty words and 62 products. Scaling evidence comes from synthetic worlds of up to 841 symbols, not from a large real vocabulary.

## What would test it further

The explainer lists the open questions. These are the experiments that would most strengthen or weaken the claims on this page, in order of how much they would tell us.

1.  **GraphLog** ([Sinha et al., 2020](https://arxiv.org/abs/2003.06560)): 57 logical worlds, each with its own rules, built by others. A model that derives its rules should re-derive them per world with no changes. This directly addresses the "worlds I built" threat.
2.  **Set-valued answers.** Spatial and temporal relation algebras, where the true answer is often "one of these three", and where published rule learners are reported to struggle.
3.  **Independent replication** on CLUTRR's original data release, so the comparison is on identical test sets.
4.  **A real, large vocabulary** with naturally occurring composition data, to see whether the audit still retires false laws when coverage is uneven.

## Reproducing everything

    pip install -e .
    export CLUTRR_DIR=/path/to/clutrr/gen_train23_test2to10   # the other datasets alongside it
    python -m bicyclic_kinship.benchmark    # all six datasets, the ablation
    python -m bicyclic_kinship.noise        # false and irrelevant input facts
    python -m bicyclic_kinship.worlds       # family trees, in-laws, depth, paths, grid
    python -m bicyclic_kinship.stress       # evidence, labels, tree seeds, refusals, cost
    pytest                                  # 59 tests

Zero dependencies, pure Python 3.10+. The synthetic algebras come first in the test file on purpose, as the guard that fails if domain knowledge ever leaks into the source. For the plain-language account, start with [*Who Is Your Daughter's Grandfather?*](clutrr-explained.html)

**References**

*Benchmark and published systems.* Sinha, Sodhani, Dong, Pineau & Hamilton (2019), CLUTRR, EMNLP, [arXiv:1908.06177](https://arxiv.org/abs/1908.06177). Minervini et al. (2020), CTP, ICML, [arXiv:2007.06477](https://arxiv.org/abs/2007.06477). Bergen, O'Donnell & Bahdanau (2021), Systematic Generalization with Edge Transformers, NeurIPS, [arXiv:2112.00578](https://arxiv.org/abs/2112.00578). Lu et al. (2022), R5, ICLR, [arXiv:2205.06454](https://arxiv.org/abs/2205.06454). Cheng et al. (2023), NCRL, ICLR, [arXiv:2303.03581](https://arxiv.org/abs/2303.03581). Khalid & Schockaert (2025), Systematic Relational Reasoning with Epistemic Graph Neural Networks, ICLR, [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/6590cb829f5ffef50050f3e5845fbb4c-Paper-Conference.pdf). Sinha, Sodhani, Pineau & Hamilton (2020), GraphLog, [arXiv:2003.06560](https://arxiv.org/abs/2003.06560).

*Kinship algebra.* Weil (1949), appendix to Lévi-Strauss, *Les structures élémentaires de la parenté*. Read (1984), An Algebraic Account of the American Kinship Terminology, *Current Anthropology* 25(4), [link](https://doi.org/10.1086/203160). Read, Fischer & Leaf (2013), What Are Kinship Terminologies, and Why Do We Care?, *Social Science Computer Review*, [doi](https://doi.org/10.1177/0894439312455914).

*Mathematics.* Lyapin (1953), first published description of the bicyclic semigroup. Clifford & Preston (1961), *The Algebraic Theory of Semigroups*, vol. 1.

*Program analysis.* Cousot & Cousot (1977), Abstract interpretation, POPL, [doi](https://doi.org/10.1145/512950.512973). Ernst, Cockrell, Griswold & Notkin (1999), Dynamically discovering likely program invariants to support program evolution, ICSE, [doi](https://doi.org/10.1145/302405.302467). Cousot, Cousot & Mauborgne (2011), The Reduced Product of Abstract Domains and the Combination of Decision Procedures, FoSSaCS, [doi](https://doi.org/10.1007/978-3-642-19805-2_31). Wang, Anderson, Dillig & McMillan (2018), Learning Abstractions for Program Synthesis, CAV, [doi](https://doi.org/10.1007/978-3-319-96145-3_22).

*Learning relations and rules.* Hinton (1986), Learning distributed representations of concepts, *Proceedings of the Eighth Annual Conference of the Cognitive Science Society*. Quinlan (1990), Learning logical definitions from relations, *Machine Learning* 5, [doi](https://doi.org/10.1007/BF00117105). Paccanaro & Hinton (2001), Learning distributed representations of concepts using linear relational embedding, *IEEE TKDE* 13(2), [doi](https://doi.org/10.1109/69.917563). Mitchell (1982), Generalization as search, *Artificial Intelligence* 18(2), [doi](https://doi.org/10.1016/0004-3702(82)90040-6). Gold (1978), Complexity of automaton identification from given data, *Information and Control*, [doi](https://doi.org/10.1016/S0019-9958(78)90562-4). Angluin (1987), Learning regular sets from queries and counterexamples, *Information and Computation*, [doi](https://doi.org/10.1016/0890-5401(87)90052-6). Power, Burda, Edwards, Babuschkin & Misra (2022), Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets, [arXiv:2201.02177](https://arxiv.org/abs/2201.02177).

*Symbolic systems and benchmark audits.* Read & Behrens (1990), KAES: An Expert System for the Algebraic Analysis of Kinship Terminologies, *Journal of Quantitative Anthropology* 2(4), [link](https://escholarship.org/uc/item/43f9d51h). Yang, Ishay & Lee (2023), Coupling Large Language Models with Logic Programming for Robust and General Reasoning from Text, Findings of ACL, [arXiv:2307.07696](https://arxiv.org/abs/2307.07696). Zhang et al. (2024), Extracting and Following Paths for Robust Relational Reasoning with Large Language Models (Path-of-Thoughts), *TMLR* 2026, [arXiv:2412.17963](https://arxiv.org/abs/2412.17963). Liu & Li (2011), On a Semi-Automatic Method for Generating Composition Tables, [arXiv:1105.4224](https://arxiv.org/abs/1105.4224). Vovk, Gammerman & Shafer (2005; 2nd ed. 2022), *Algorithmic Learning in a Random World*, Springer, [doi](https://doi.org/10.1007/978-3-031-06649-8).

*Knowledge-graph embeddings.* Song, Luo & Huang (2021), Rot-Pro: Modeling Transitivity by Projection in Knowledge Graph Embedding, NeurIPS, [arXiv:2110.14450](https://arxiv.org/abs/2110.14450). Yang, Wang, Sha, Engelbrecht & Hong (2022), Knowledgebra: An Algebraic Learning Framework for Knowledge Graph, [arXiv:2204.07328](https://arxiv.org/abs/2204.07328).


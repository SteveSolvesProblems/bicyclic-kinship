

CLUTRR · kinship reasoning · bicyclic-kinship

# Kinship Reasoning Without a Neural Network

By **Steve Ball** · September 2026

Who is your daughter's grandfather?

Your father, or your spouse's father. The words alone can't say which. This model says so: it answers **father or father-in-law**, and then tells you which one the evidence favours.

A plain-language explainer of an algebraic model on the CLUTRR benchmark.

*\[figure — see the HTML version\]*

**Two routes, one question.** Your daughter's grandfather is two steps up from her. Through you, that's your father. Through your spouse, it's your spouse's father. The chain *(daughter, grandfather)* doesn't say which route, so the model answers **father or father-in-law**.

**How this was built.** Built with Claude Code, using Claude Opus models. The research direction and the claims are the author's. The record below is evidence that this is an ongoing line of research, not a one-off.

- **8** commits of development, Sep 3–30, 2026, published as one reference snapshot
- **~780** commits in two earlier research repositories it grew from, since July 2026
- **59** tests; every number on this page is checked by a test or printed by a script
- **~2,000** lines of Python, zero dependencies
- **2** writeups: this explainer and a technical companion
- **13** published revisions of this page
- **11** claims withdrawn or corrected, on the record

The eleven corrections

1.  A gender result that depended on listing word pairs male-first was withdrawn: domain knowledge had been smuggled in through a variable name.
2.  An early claim that comparing against published systems was meaningless was too modest. They receive the same symbolic input, so the comparison stands.
3.  "A family of four" coordinate systems was a limit in the search code, not a fact about the data.
4.  A proposition presented as new turned out to be known in the knowledge-graph literature. The paper now says so.
5.  The method was first framed as verification. It is static analysis, and the writeups now name that lineage.
6.  A claim that the kinship literature calls this structure a "bicyclic semigroup" had no source we could find. It was replaced with Read's own equations.
7.  Published in-law numbers came from one run of a tie-break that varied between runs. The tie-break is now fixed and the numbers are deterministic.
8.  The in-law "structural limit" was narrowed twice: first by training on longer chains, then by splitting words with two meanings (see [two extensions, tested](#extensions)).
9.  An earlier writeup said that under noisy labels the model "stays sound throughout". That was measured only on the words it still knew. Noisy labels also delete words, and the claim now says so.
10. Several stress-test tables had no script behind them. A new script re-measured them, and two sets of numbers changed.
11. This page said every published CLUTRR system is trained. One is instead given hand-written rules. Found while comparing the claims against prior work.

Contents

1.  [The short version](#short)
2.  [Why this matters](#why)
3.  [Questions and answers](#examples)
4.  [What this is, and what it isn't](#scope)
5.  [Not statistical machine learning](#not-ml)
6.  [What "opaque symbols" means](#opaque)
7.  [A model that chooses its own rules](#kind)
8.  [The four rule shapes](#shapes)
9.  [The same algebra, built by hand in 1984](#history)
10. [Results on all six CLUTRR datasets](#results)
11. [What the model noticed about CLUTRR itself](#noticed)
12. [Why soundness matters](#sound)
13. [How deep can it go?](#depth)
14. [Where this approach stops](#limits)
15. [Making it more general](#general)
16. [Two extensions, tested](#extensions)
17. [Why derived laws matter for new inputs](#novel)
18. [Related work](#company)
19. [Open questions](#open)
20. [Putting it together](#close)
21. [Reproducing everything](#repro)

## The short version

CLUTRR ([Sinha et al., 2019](https://arxiv.org/abs/1908.06177)) is a benchmark that asks questions like *"Alice's father's father's daughter is Alice's what?"* (Her aunt.) It's easy to train a system on short chains of relations like that. It's hard to get that system to answer longer chains than it was trained on. Statistical machine learning is known to struggle with that step.

This repository answers those questions with no neural network, no trained weights and no random seeds. It is given the relation words as **opaque symbols**, strings whose meaning it never sees, plus a few thousand short examples. From those it **derives** the rules by which the symbols combine. Then it applies those rules to chains of any length.

On CLUTRR's standard "train short, test long" splits it scores at the level of the best published rule-learning systems, and it is **sound** on every test story. Sound means the correct answer is always inside the set of answers it gives.

## Why this matters

I wanted a reasoning system that can explain itself in the strong sense. Most explanations from AI systems today are a story told about an answer after the fact: a heat map over the input, or a paragraph of reasoning generated alongside the output. If the story is wrong, nothing breaks, because there is nothing underneath it to check it against. The other kind of explanation is a *derivation*, like long division or a proof. The steps are the computation itself, and you can check each one.

Relational reasoning, chaining facts together to get a new one, is where story-style explanations are normal today. Kinship is the smallest version of it that can't be faked. Nobody memorizes the answer to a ten-step chain. You either combine the steps correctly or you don't, and anyone can check.

**The point.** On problems that have a composition law, you can derive the law instead of learning it. The derived model reaches the accuracy of the best trained systems, and it gives you things they can't:

- **Answers you can audit.** Ask whether an answer is guaranteed or merely likely, and you get different words back.
- **Honest uncertainty.** When the facts don't settle a question, like the daughter's grandfather, it says so instead of picking one and sounding sure.
- **Length for free.** A rule applies to a thousand-step chain exactly as it applies to a two-step one.
- **Failures you can read.** When it's wrong, the wrong rule is something you can inspect, disagree with and test.

The default in this field is to learn. Every published CLUTRR system I have found either learns its rules by training or is given them by hand, as a logic program ([Yang, Ishay & Lee, 2023](https://arxiv.org/abs/2307.07696)). None derives them. Meanwhile the mathematics that solves it, the algebra of relations and the program-analysis theory behind sound answers, has been mature for decades, and it runs inside production tools every day. The two literatures rarely talk to each other. This project connects them.

What I don't know yet is how much of reasoning has a law like this. Kinship has one. Filesystem paths and grid movement have one. Clock arithmetic has one too, but it's a shape this model doesn't have yet, and the model says so rather than forcing a fit. The method is cheap enough to point at a new domain and ask. If most things have a law, that's a large result. If only a few tidy corners do, it's a small one. Finding out is the work.

## Questions and answers

Each question reaches the model as a chain of relation words. It answers with every word its rules allow. When that's more than one word, it also says which one the training data favours. These are real outputs of the model trained on CLUTRR.

- Your mother's son (mother, son) **brother** \[One answer\]
- Your father's father's daughter (father, father, daughter) **aunt** \[One answer\]
- Your son's wife (son, wife) **daughter-in-law** \[One answer\]
- Your daughter's grandfather (daughter, grandfather) **father or father-in-law** \[Two possible\] It depends on which of your daughter's parents you go through. The model admits both and prefers *father*, the label CLUTRR's training data used 587 times against 227 for *father-in-law*.
- Your wife's son's grandmother (wife, son, grandmother) **mother or mother-in-law** \[Two possible\] This exact chain appears in CLUTRR's test set with *both* labels. No method that reads the chain can get every copy right.
- Your mother's brother's son (mother, brother, son) **no answer** \[No word\] He's your cousin. CLUTRR's twenty words don't include *cousin*, so the model returns nothing instead of the nearest guess.
- Your husband's wife (husband, wife) **no answer** \[No word\] That's you. CLUTRR has no word for "self".
- A ten-step chain from CLUTRR's test set (daughter, sister, son, aunt, daughter, father, sister, husband, daughter, sister) **granddaughter** \[One answer\] Matches CLUTRR's label. The model was trained only on chains of two and three steps.
- Your mother's husband (mother, husband) **no answer** \[Miss\] The truth is your father. CLUTRR never shows this combination, and it lets a wrong rule survive that blocks the answer. Trained on a real family tree instead, the model answers *father*. See [what the model noticed about CLUTRR](#noticed).

## What this is, and what it isn't

Four limits come first, because each one matters.

- **Tested mainly on kinship, with four kinds of rule.** The model is general-purpose, but the evidence here is mostly kinship, plus a few small non-family worlds. It searches four simple rule shapes, described below. The shapes are chosen by a person, which is a real prior and is stated as one. What the model derives from data is *which* rules of those shapes hold, and every number inside them.
- **The input is already symbolic.** The model reads relation chains such as `(mother, brother, son)`, not stories in English. Turning prose into chains is a separate, hard problem that this work does not attempt. The published systems compared below are given the same symbolic input.
- **The vocabulary is given.** Someone supplies the twenty relation words. Deciding what the relations are in the first place is not addressed.
- **Soundness is checked, not proved.** Every derived rule is tested against all the training data, and any rule that fails is thrown out. A rule that survives can still be wrong about a case the data never shows. That is the same status as the "likely invariants" that program-analysis tools such as Daikon infer from test runs ([Ernst et al., 1999](https://doi.org/10.1145/302405.302467)). There's an example of exactly that below.

## Not statistical machine learning

Most systems evaluated on CLUTRR are neural networks: graph networks, recurrent networks, transformers. The strongest ones are hybrids, networks trained to find logical rules, such as CTP ([Minervini et al., 2020](https://arxiv.org/abs/2007.06477)), R5 ([Lu et al., 2022](https://arxiv.org/abs/2205.06454)) and NCRL ([Cheng et al., 2023](https://arxiv.org/abs/2303.03581)). All of them tune millions of numbers by gradient descent or reinforcement learning, and report the average over several runs with different random seeds.

This model does none of that. All it learns is a small table of whole numbers, derived with exact fractional arithmetic. Nothing is sampled, nothing is approximated, and running it twice gives the same answer to every decimal place. Training takes a fraction of a second.

**Why statistical ML finds CLUTRR hard.** A network trained on chains of two or three steps learns what two-to-three-step chains *look like*. Ask it about a ten-step chain and it is outside anything it has seen. The CLUTRR authors built the benchmark to expose exactly this ([Sinha et al., 2019](https://arxiv.org/abs/1908.06177)), and the published numbers show it.

Trained on 2–3 step chains. Published numbers from R5 ([Lu et al., 2022](https://arxiv.org/abs/2205.06454)), Table 2.

| System               | Kind                 | 4 steps | 10 steps |
|----------------------|----------------------|---------|----------|
| GCN                  | graph neural network | 84%     | 39%      |
| GAT                  | graph neural network | 91%     | 45%      |
| Multi-head attention | transformer-style    | 81%     | 67%      |
| LSTM                 | recurrent network    | 98%     | 75%      |

*\[figure — see the HTML version\]*

**Longer chains than it was trained on.** Every point is a chain longer than anything in training. The trained networks decay as chains grow, because length is something they had to learn. The model's line stays flat, because length never enters its calculation. Published numbers from R5 (Lu et al., 2022), Table 2.

**Why this model doesn't have that problem.** It never learns what chains look like. It learns how two relations *combine*, and a ten-step chain is just nine of those combinations, one after another. Length never enters the calculation, so there's nothing to degrade.

## What "opaque symbols" means

The model's input is a list of facts of the form *"symbol A followed by symbol B gives symbol C."* On CLUTRR there are 62 such facts, taken from the two-step training chains. For all the model knows, the symbols could be `x17`, `x4` and `x9`.

That claim can be checked in the code:

- The derivation and solving code (`derive.py`, `cone.py`, `projection.py`, `system.py`, `solver.py`, `pathrules.py`) contains **no kinship words** except in comments. It has no list of genders, no idea of "parent", and no special handling of "in-law".
- The same code, unchanged, runs on things that aren't families at all: a stack of push and pop operations, filesystem paths with `..`, movement on a grid, and several small algebras. Those tests fail if kinship knowledge ever leaks into the code.
- When the data fits none of the four rule shapes, as with the six symmetries of a triangle, the model derives **nothing** and says it doesn't know. It does not force a fit.

## A model that chooses its own rules

This is a model, even though there's no neural network in it. What makes it one is what it does with data.

A rule engine starts with rules a person wrote and applies them to the input. This model starts with no rules about the domain at all. It has a small menu of rule *shapes* (a counter, a stack, and two kinds of classification, all explained [below](#shapes)), and it looks at the data to decide which of them apply:

- It **keeps every shape the data fits**, and drops any shape the data contradicts, even once.
- It **derives the specific rules** inside each shape it keeps, every number of them, from the data.
- It **re-derives as the data grows.** With a quarter of CLUTRR's facts, wrong rules survive because nothing contradicts them yet. With all of them, those rules are retired and the right ones are pinned down.

*\[figure — see the HTML version\]*

**62 facts in, 190 answers out.** Each cell is one pair of words, such as "mother, then son". Training showed the model 62 of the 400 pairs. The rules it derived answer 128 more, with no lookup table. The 210 empty cells are pairs CLUTRR's twenty words have no name for, such as a cousin, and the model returns nothing there instead of guessing.

Nothing in that process knows it's looking at families. Give it a different dataset whose relations combine in one of these shapes and it derives that dataset's rules instead. The tests do exactly that, with the same code, on filesystem paths, movement on a grid and several small algebras. That's what makes it general-purpose: the domain comes from the data, not from the code.

"Fits" here means fits *exactly*. A shape is kept only if it explains every observed fact, not most of them. That's what makes the answers checkable, and it's also why noisy data is a weakness (see [where this approach stops](#limits)).

## The four rule shapes

A relation word such as `grandmother` gets a few coordinates, like an address. Each rule shape says how the address of a combined relation follows from the addresses of its parts. The model searches all four shapes and keeps whichever ones the data supports.

1.  **A counter.** Some quantity simply adds up along the chain. For kinship it turns out to be *generations*: up one for a parent, down one for a child, zero for a sibling or spouse. Nobody told it that. It is the only additive quantity the 62 facts allow.
2.  **A stack, where up-then-down cancels.** This is the important one. Your mother's son is your *brother*, not a "mother-son": going up and then down partly cancels. A plain counter can't express that, but a stack can. Think of a browser's back button, or `..` in a file path: going into a folder and then back out leaves you where you started. The model finds that each kinship word is some number of steps up followed by some number of steps down:
    - father↑(1,0)
    - son↓(0,1)
    - brother↑↓(1,1)
    - grandfather↑↑(2,0)
    - uncle↑↑↓(2,1)
    - nephew↑↓↓(1,2)
    - husband·(0,0)

    Mathematicians call this structure the *bicyclic monoid*, first described by Lyapin in 1953. An anthropologist built the same structure for kinship by hand in 1984 ([Read, 1984](https://doi.org/10.1086/203160); see [the next section](#history)). The model re-derived it from 62 facts about symbols it can't read.
3.  **"The last step decides."** Some properties come only from the final word of the chain. Gender is the obvious one: whoever you pass through, a chain ending in `mother` names a woman. The model finds six such classes. It is never told the word "gender".
4.  **"The first step decides."** This is the mirror image of shape 3. The model searches for it and finds that kinship has no such property, so it is dropped. Nobody declares that in advance.

*\[figure — see the HTML version\]*

**Where each word lives, as derived.** Rows count steps up, columns steps down. Nobody placed these words: this layout is what the model derived from 62 facts about opaque symbols. The shaded cell is the in-law limit. *Father* and *father-in-law* share one address, and no rule separates them. (Son and son-in-law also share a cell, but the "last step decides" rule tells them apart.) Two up and two down is a real place with no CLUTRR word. That's where a cousin would be.

To answer a question, the model follows each chain through every rule that survived, which gives an address. It then returns **every** word that sits at that address. Usually that is one word. Sometimes it is two, and when that happens it is almost always telling the truth (see [what it noticed about CLUTRR](#noticed)).

What each rule shape is worth, measured by switching it off (on `gen_train23`).

| Configuration                   | Top-1 accuracy |
|---------------------------------|----------------|
| Everything                      | 99.4%          |
| Without the stack               | 72.3%          |
| Without "the last step decides" | 87.4%          |
| Without the counter             | 98.7%          |

The stack is the largest single contributor here, and it is what lets most answers be pinned to a single word. The "last step" rule carries every male/female distinction. Without it, no answer is ever narrowed to one word. The counter is almost redundant once the stack exists, because the stack already contains it: steps down minus steps up is the generation count.

## The same algebra, built by hand in 1984

Symbolic methods stand on established mathematics, and this one is no exception. People have been writing kinship down as algebra for more than 75 years. André Weil worked out the algebra of marriage rules for Claude Lévi-Strauss in 1949. Harrison White's *An Anatomy of Kinship* (1963), [Boyd, 1969](https://doi.org/10.1016/0022-2496(69)90032-7) and [Lorrain & White, 1971](https://doi.org/10.1080/0022250X.1971.9989788) treated kinship and other social ties as algebras in which relations compose, the way "my mother's brother" composes two relations into one.

The closest match is Dwight Read's algebra of the American kinship terminology ([Read, 1984](https://doi.org/10.1086/203160)). Read built it by hand, knowing what every word means. He defined a *kin term product* ("father of mother is grandfather"), took parent and child as generating terms with spouse added later, and wrote down the structural equations that generate the whole terminology. It was later implemented as software, the Kinship Algebra Expert System ([Read, 2006](https://doi.org/10.1177/0894439305282372)).

The model derived the same structure from 62 facts about symbols it can't read:

| Read's structural equation | What the model derived | Match |
|----|----|----|
| parent of child = self | child, then parent, lands on (0,0): no steps up or down | Yes |
| child of parent = sibling | parent, then child, lands on (1,1): brother or sister | Yes |
| spouse is a separate generator | husband and wife also land on (0,0), beside self | Partly |

Both equations are quoted in [Read, Fischer & Leaf, 2013](https://doi.org/10.1177/0894439312455914). The first one is the whole definition of the bicyclic monoid: in mathematics it is written *pq = 1* ([Lyapin, 1953](https://en.wikipedia.org/wiki/Bicyclic_semigroup)). Read's version reads it with *p* as "parent of" and *q* as "child of". The second equation is the other order, which does not cancel. That one-sided cancellation is exactly the "stack" shape the model chose.

The third row is where the derived model is weaker than the hand-built one. Read treats marriage as its own generator. The model puts spouses at the same point as self, because nothing in CLUTRR separates them. That is the in-law limit described [further down](#limits).

Matching 1984 is the result, not a lack of novelty. A method that recovers, from a few dozen data points, a structure a field established by hand has found something real, not something that merely fits. The claim here is the derivation, not the algebra.

## Results on all six CLUTRR datasets

The public CLUTRR release has six datasets, and all six are reported here. They fall into two groups that test different things.

**The two extrapolation splits.** Train on short chains and test on longer ones. These are what the benchmark exists for.

| Dataset | Trained on | Tested on | Stories | Sound | Top-1, unseen lengths |
|----|----|----|----|----|----|
| `gen_train23_test2to10` | 2–3 steps | 4–10 steps | 1,003 | 100% | **99.80%** |
| `gen_train234_test2to10` | 2–4 steps | 5–10 steps | 826 | 100% | **99.39%** |

Per chain length, next to the best published systems. Published numbers are averages over several runs with ±1–5% spread. This model is deterministic and has no spread.

Trained on 2–3 steps. Published numbers from R5 ([Lu et al., 2022](https://arxiv.org/abs/2205.06454)), Table 2.

| Steps                       | 4    | 5    | 6   | 7   | 8   | 9   | 10  |
|-----------------------------|------|------|-----|-----|-----|-----|-----|
| This model                  | 99.5 | 99.4 | 100 | 100 | 100 | 100 | 100 |
| R5 (ICLR 2022)              | 98   | 99   | 98  | 96  | 97  | 98  | 97  |
| CTP<sub>A</sub> (ICML 2020) | 99   | 99   | 99  | 96  | 94  | 89  | 90  |

Trained on 2–4 steps. Published numbers from EpiGNN ([Khalid & Schockaert, 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/6590cb829f5ffef50050f3e5845fbb4c-Paper-Conference.pdf)), Table 1.

| Steps              | 5    | 6   | 7    | 8   | 9   | 10  |
|--------------------|------|-----|------|-----|-----|-----|
| This model         | 98.4 | 100 | 98.7 | 100 | 100 | 100 |
| NCRL (ICLR 2023)   | 100  | 99  | 98   | 98  | 98  | 97  |
| R5 (ICLR 2022)     | 99   | 99  | 99   | 100 | 99  | 98  |
| EpiGNN (ICLR 2025) | 99   | 99  | 99   | 99  | 96  | 98  |

Read this as **the same level as the best published systems**, not better. The top systems sit within their own error bars of 100%. Also, the published papers use CLUTRR's original data release, while these runs use the HuggingFace release of the same splits. The two are generated the same way, but the story counts differ slightly, so they are not identical test sets. The claim that holds up: a model with no network and no weights, reading opaque symbols, reaches the level of trained neuro-symbolic rule learners, and it says when it is unsure instead of guessing.

**The four robustness splits** (`rob_*`: clean, disc, irr, sup). CLUTRR built these to test distracting *sentences* in the prose. This model never reads prose, so for it these are four more independently generated samples. They only contain chains of 2–3 steps (nine distinct chains per test set), so they test consistency, not extrapolation. The model was trained once on `gen_train23` and never retrained.

| Dataset           | Stories | Sound | Top-1   | Best possible |
|-------------------|---------|-------|---------|---------------|
| `rob_train_clean` | 447     | 100%  | 100.00% | 100.00%       |
| `rob_train_disc`  | 445     | 100%  | 94.83%  | 94.83%        |
| `rob_train_irr`   | 444     | 100%  | 93.69%  | 93.69%        |
| `rob_train_sup`   | 447     | 100%  | 93.74%  | 93.74%        |

"Best possible" is explained in the next section. The model derives the identical set of rules on every dataset, down to the last coordinate, whether it is trained on that dataset or not.

## What the model noticed about CLUTRR itself

Because the model returns *every* answer its rules allow, it points to questions where the chain doesn't contain the answer.

**Some CLUTRR questions have two right answers, and the dataset marks one wrong.** Take *wife → son → grandmother*. Your wife's son is your son, and your son's grandmother is **either** your mother **or** your wife's mother. The chain alone can't say which, because both are real people it could be. CLUTRR has one correct label per story, and the same chain appears in the test set with *both* labels. In every one of the six datasets, every such case is *mother* vs *mother-in-law* or *father* vs *father-in-law*.

That puts a hard ceiling on any method that answers from the chain. In the worst case (`rob_irr`) **6.3% of test stories can't be scored correct by any such method**. The model's two-word answer on those questions is correct; the benchmark just can't score it. The "best possible" column above is that limit. On all four `rob_*` datasets the model sits exactly on it, and on the two extrapolation splits it is within 0.2% and 1.4% of it.

These are not labelling mistakes. The labels are true of the family trees CLUTRR generated. The point is narrower: the symbolic version of the benchmark asks some questions its input can't settle. Anyone can check this without trusting the model. Group the test set by chain, and count the chains with more than one label.

Others have spotted individual cases by reading stories. Yang, Ishay & Lee ([2023](https://arxiv.org/abs/2307.07696)) found 16 faulty instances in a 400-story CLUTRR sample, one of them labelled *mother* where *mother-in-law* was right, and Zhang et al. ([2024](https://arxiv.org/abs/2412.17963)) note questions with several correct answers but one label. What the model adds is the exact count on all six datasets, found from the derived rules rather than by inspection, and the finding that every such case is a blood/in-law pair.

**CLUTRR's coverage lets a false rule pass every test.** In the entire corpus, every chain ending in *husband* has the answer *son-in-law*, because the only such chain that ever occurs is *daughter → husband*. So "the last step decides" happily files *husband* with *son-in-law*, and nothing in CLUTRR can contradict it. Run the model on a real, generated family tree instead and that rule is visibly false: your mother's husband is your father. Soundness there drops to 79–90%. Train on the tree's own data and the rule corrects itself, returning to 100%. Along the same lines, no in-law word ever appears *inside* a CLUTRR chain, only as an answer, so nothing in the benchmark shows how in-law relations combine.

The lesson generalizes: **a benchmark's coverage decides which rules can be learned from it, and no amount of checking against that benchmark can see past it.**

## Why soundness matters

Returning all twenty words would be 100% sound and useless. The claim is soundness *with an average answer of 1.1 words*. Most answers are one word, and the rest are the genuinely ambiguous cases above.

The same property holds when the input is wrong. Add false facts to the story, so there are two routes between the people that disagree, and the model combines the routes by keeping only answers both allow. Two routes that disagree leave *nothing*, and an empty answer is a flagged contradiction. With ten false facts per story it flags 98% of stories and gives a confidently wrong answer on **none**. A system that returns a single best guess has nothing to disagree with.

This notion of soundness is borrowed, not invented. Each rule shape is a safe over-approximation of the truth, and answering with the overlap of several of them is the core idea of *abstract interpretation* ([Cousot & Cousot, 1977](https://doi.org/10.1145/512950.512973)), the theory behind the static analysers that check production software.

## How deep can it go?

CLUTRR stops at ten steps. How far can this go? A thousand steps? Ten thousand?

The model doesn't care. A chain is folded one step at a time, so a 10,000-step chain is just more of the same arithmetic. The hard part is finding **true answers** to check it against, and that turns out to be a fact about families, not about the model.

**Why real family trees stop at about twenty steps.** A fair question never visits the same person twice. Once it does, the chain of words no longer decides the answer. (Your father's grandson is your son if he's your own child, and your nephew otherwise.) Any real family tree is finite, so a walk of about twenty steps runs out of new people. That's the same reason CLUTRR stops at ten.

**So the test grows the family as it walks.** Nobody exists until the walk needs them: parents are created the first time someone asks for one, and a new child can always be born. Walks of any length never meet anyone twice, and the true answer is still read off the family tree by counting generations to the nearest common ancestor. It is never worked out by combining the words.

Rules derived from CLUTRR (chains of 2–3 steps, never retrained), on walks that only pass through relatives CLUTRR has words for.

| Chain length | Chains | Sound | Top-1 | Answer size | Time per answer |
|--------------|--------|-------|-------|-------------|-----------------|
| 10           | 300    | 100%  | 100%  | 1.00        | 0.07 ms         |
| 100          | 300    | 100%  | 100%  | 1.00        | 0.28 ms         |
| 1,000        | 100    | 100%  | 100%  | 1.00        | 2.9 ms          |

Every answer at every length is the single correct word. That covers a thousand steps, from rules learned on chains of three.

**Past the edge of the vocabulary.** Let the walk wander freely and after 10,000 steps it lands on someone English has no word for, typically a couple of generations up and ten thousand down. The model correctly returns *no name*. Underneath, though, it still computes an address, and that address can be checked directly. On every walk tested, to 10,000 steps, the derived coordinates equal the true (generations up, generations down) exactly. The model knows precisely where that person sits in the family. It just has no word to call them.

For the curious: the CLUTRR data leaves four candidate coordinate systems standing. One is the anthropologists' system. The other three turn out to be the same system shifted by exactly one generation up and one down. That was measured, not assumed.

## Where this approach stops

Every limit below is measured in the repository, not guessed.

- **Rules that wrap around.** Clock arithmetic (11 o'clock plus 3 hours is 2 o'clock) and the six symmetries of a triangle fit none of the four shapes. The model derives nothing and answers "any of them". That is sound, but useless. It never forces a fit.
- **In-law questions.** *Father* and *father-in-law* are one generation up either way, and on CLUTRR the model keeps both wherever the chain genuinely doesn't decide. On a family tree with marriages, the rules alone first pinned only 9–35% of answers to one word. Most of that gap turned out to be English words with two meanings, and splitting them closes it (see [two extensions, tested](#extensions)).
- **Rules are only as good as the data that could disprove them.** Give the model 15 of its 62 facts instead of all 62 and, on the questions it attempts, the right answer is missing from its answer about 90% of the time, with no warning. A rule fitted to too little evidence looks exactly as exact as a true one. The *husband* example above is the same failure in the wild.
- **Noisy labels.** Exact derivation is brittle. Corrupt one training answer in two hundred and every rule gets thrown out, so the model abstains instead of guessing. Corrupt one in a hundred and it also starts losing words: a word whose training answers disagree drops out of its vocabulary, and questions that use it can't be answered at all. An opt-in tolerance setting recovers everything up to 10% corrupted labels and gives up at 20%.
- **Worlds richer than the rules.** Filesystem paths with two folder names (`x/y` is not `y/x`) have the same up/down shape but more going on. The model stays 100% sound, and its answers widen to about six candidates instead of one.
- **Someone has to supply the symbols.** Turning raw text or images into relation symbols in the first place is the harder half of the problem, and this work doesn't touch it.
- **Scale is fine, not free.** It uses exact fractions, not floating point. Deriving the rules for 841 symbols takes under half a minute on a laptop, and answering is fast at any length.

Depth isn't on this list. As the previous section shows, it isn't a limit.

## Making it more general

The method is a recipe that has nothing to do with families:

1.  Treat the relation words as opaque symbols, and collect observed combinations (*A then B gives C*).
2.  Keep a menu of rule shapes, each with an exact way to derive its numbers from data.
3.  Check every derived rule against *all* the data, and throw out any rule that fails once.
4.  Answer by combining the surviving rules, and return every answer they allow.

Generalizing means growing step 2's menu, or finding new places where the existing menu fits.

**Shapes worth adding.** These are the natural next ones, each aimed at a limit above. The last two have been built and tested; the results follow.

- *Wrap-around counters* (clock or modular arithmetic), for cyclic structure such as days of the week or rotations.
- *Small symmetry groups*, where order matters and there is no single number line, such as rotating and flipping a shape.
- *"Take the larger" rules*, for things like deadlines or permission levels, where combining two values keeps the bigger one.
- *Answers that are sets.* In spatial reasoning ("A is inside B, B overlaps C, so A is…?") and in time-interval reasoning, the true answer is often "one of these three". The EpiGNN paper ([Khalid & Schockaert, 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/6590cb829f5ffef50050f3e5845fbb4c-Paper-Conference.pdf)) reports that neuro-symbolic rule learners like R5 and NCRL are "largely ineffective" there. This model already returns sets, so that is a natural place to test it next.
- *Words with more than one meaning*: let one word stand for several addresses, split from the data.
- *A route-memory lane*: a small rule that remembers something about the route taken, such as "crossed a marriage". Learning the smallest such state machine from examples is a long-studied problem called grammatical inference ([Gold, 1978](https://doi.org/10.1016/S0019-9958(78)90562-4); [Angluin, 1987](https://doi.org/10.1016/0890-5401(87)90052-6)).

**Places the existing shapes already fit.** Two are tested here: filesystem paths (`..` cancels a folder, the same up/down structure as kinship, exact to 10,000 steps) and movement on a grid, where the model finds it needs *two* counters, not one. Other domains share the up/down "stack" shape but are **untested**: undo/redo histories, matching brackets and nested function calls, and network packets that are wrapped and unwrapped in layers. Physical units have the "counter" shape, since combining metres per second with seconds adds the exponents.

## Two extensions, tested

Both are optional and off by default, and neither changes any CLUTRR number on this page. Both passed the same controls as the core model: on the synthetic worlds that lack the structure they look for, they derive nothing.

### Words with more than one meaning

Real vocabularies reuse words. In English, "first cousin once removed" means both your parent's cousin and your cousin's child, and "brother-in-law" means both your spouse's brother and your sister's husband. One word at two addresses breaks the counter and the stack for the whole vocabulary. With English-style cousin names on a generated family tree, the model derived no counter and no stack at all, and top-1 fell to 31% at two steps. It stayed sound, but it was nearly useless.

The fix is a domain-free rule. A word that only ever appears as an answer may split into the fewest *senses* its facts require, each with its own address. The model still answers with the surface word. A chain containing an ambiguous word is composed once per sense, so splitting can only widen an answer. It never drops a true one.

- **Cousins and removes, named the English way.** "First cousin" splits into two senses and "first cousin once removed" into four. Each sense matches exactly one true relationship on the family tree. Soundness is 100%, and every answer is pinned to the right single word from 2 to 10 steps.
- **In-laws.** On a family tree with marriages, six in-law words split into English's two meanings, found from the data alone. For *brother-in-law* the two senses are exactly "sister's husband" and "spouse's brother". Answers the rules alone pin to one word go from 9–35% to 100%, still 100% sound, across nine combinations of training and test trees.
- **A non-family check.** On a grid where one word means both north-east and south-east, the split recovers the second counter and every answer is exact.

Two caveats. The family-tree walks never pass back through anyone, so every answer there is fully determined. On CLUTRR, where "your daughter's grandfather" genuinely has two answers, nothing splits and both answers stay. And a word that is ambiguous *and* used inside chains, not just as an answer, isn't handled yet.

### Route memory

The second extension is a small state machine that updates step by step along a chain. It is found by searching for the smallest one that fits every training example exactly, then audited and combined like the other shapes.

- **It finds real structure without being told.** On the family tree with marriages, the smallest machine it found splits the vocabulary exactly at "-in-law", a string the code never reads. On a non-family world built to need it (a switch that steps can set, clear or leave alone), it derives exactly three states and is exact out to 100 steps.
- **It helped less than expected.** It raised the share of in-law questions settled to one word from 9% to 24% at two steps, and barely at five. The senses split, above, turned out to be the real fix.
- **The audit earned its place.** A more expressive variant overfit on thin data and became unsound (65% at five steps). Given longer training chains, the audit caught it and threw it out. On CLUTRR, the audit retired the route rule on 52 contradicting training chains.

The in-law problem was smaller than it first looked, and different. It was mostly two words hiding in one. That came from testing the claim instead of assuming it.

## Why derived laws matter for new inputs

This is the heart of the argument, and the evidence is now in, so here it is plainly.

A statistical model learns a function from examples. It is most reliable where examples are dense, which is *interpolation*. Ask it about something outside what it saw, such as a chain three times longer than any in training, and it is *extrapolating*. Nothing inside it states what it will do there. It might be right, and some learned systems are remarkably good at this. But you find out one input at a time, after the fact, and the model can't tell you in advance which inputs it is sure about. Chollet argues that intelligence should be measured by how well a system handles situations it wasn't prepared for, not by its skill on familiar ones ([Chollet, 2019](https://arxiv.org/abs/1911.01547)).

A derived law is different in kind. "Up one, then down one" is a claim about *every* chain, including ones nobody has seen. The results above show what that buys:

- **You can audit it before you trust it.** The whole model is a few rules you can read. You can check them against any data you have, or against a family tree, and a single counterexample falsifies one. That is how the *husband* problem was found.
- **New inputs are old rules applied again.** A 1,000-step chain isn't out of distribution for a law, because the law never had a distribution. That's why the depth results stay at 100%.
- **It knows the edge of what it knows.** When the rules admit two answers, or none, the model says so. A statistical model gives you a confidence score, which is a guess about its own reliability. The model here gives you a set, which is guaranteed to contain the truth whenever its rules are true.

Derived laws aren't automatically right. A law can fit all your data and still be false in the world, like the *husband* rule. It's also less forgiving of noisy labels than a statistical model. But when it's wrong, it's wrong in a way you can see, point to and fix with data. That's what "auditable" means here.

## Related work

Deriving explicit laws from data isn't new, and this work belongs to a long line of systems that do it. A few neighbours, and how they differ:

- **Laws of physics from measurements.** Symbolic regression recovers equations of motion from experimental data ([Schmidt & Lipson, 2009](https://doi.org/10.1126/science.1165893)), sparse regression discovers the differential equations governing a system ([Brunton, Proctor & Kutz, 2016](https://doi.org/10.1073/pnas.1517384113)), and AI Feynman rediscovered all 100 equations in its Feynman-lectures benchmark ([Udrescu & Tegmark, 2020](https://doi.org/10.1126/sciadv.aay2631)). They search over formulas; this model searches over composition rules.
- **Logic programs from examples.** Inductive logic programming learns rules like "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)" from examples, including invented helper predicates ([Muggleton et al., 2015](https://doi.org/10.1007/s10994-014-5471-y)) and by learning from failed hypotheses ([Cropper & Morel, 2021](https://doi.org/10.1007/s10994-020-05934-z)), which resembles this model's audit. A differentiable version trades exactness for tolerance of noise ([Evans & Grefenstette, 2018](https://doi.org/10.1613/jair.5714)).
- **Programs and their building blocks.** DreamCoder learns a library of reusable program pieces while solving tasks ([Ellis et al., 2021](https://doi.org/10.1145/3453483.3454080)).
- **State machines from sequences.** Grammatical inference learns automata from examples ([Gold, 1978](https://doi.org/10.1016/S0019-9958(78)90562-4); [Angluin, 1987](https://doi.org/10.1016/0890-5401(87)90052-6)), which is what the route-memory prototype does.
- **Invariants from program runs.** Daikon proposes candidate invariants and keeps the ones no execution contradicts ([Ernst et al., 1999](https://doi.org/10.1145/302405.302467)), the same audit, with the same "likely, not proved" status.

What this model adds is narrow: exact derivation over a fixed menu of *composition* shapes, sound set-valued answers, and a direct comparison against trained systems on a benchmark built to test extrapolation.

## Open questions

The introduction asked how much of reasoning has a law like this. No survey I know of answers it directly. The nearest are a general framework for the relation algebras used in qualitative spatial and temporal reasoning ([Ligozat & Renz, 2004](https://doi.org/10.1007/978-3-540-28633-2_8)) and a review of what "generalization" means across NLP research ([Hupkes et al., 2023](https://doi.org/10.1038/s42256-023-00729-y)). It breaks into smaller questions that can each be tested.

- **Which other domains have these shapes?** Good candidates, all untested: undo/redo histories, matching brackets and nested function calls, network packets wrapped and unwrapped in layers, commit ancestry in version control (until merges appear), reporting lines in an organisation, and units of measure, where exponents add. Each is cheap to try, and the model says plainly when a domain doesn't fit.
- **Can it handle a benchmark where every world has different rules?** GraphLog ([Sinha et al., 2020](https://arxiv.org/abs/2003.06560)) gives 57 logical worlds, each with its own rules. A model that derives its rules from data should re-derive per world with no changes. That's a direct test of the general-purpose claim.
- **What about answers that are genuinely sets?** Spatial and time-interval reasoning often have "one of these three" as the true answer, and published rule learners struggle there ([Khalid & Schockaert, 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/6590cb829f5ffef50050f3e5845fbb4c-Paper-Conference.pdf)). The model already answers in sets.
- **Can the model propose its own shapes?** Shapes combine by intersection, so each new one multiplies what the model can tell apart. Senses made the existing shapes work again on in-law words, which none of them could do alone. But today a person writes the menu, and the derivation for each shape. A fully general version would propose new shapes and their derivations too, which starts to look like program synthesis over algebraic structures. The risk is that a big enough menu becomes a hiding place for domain knowledge. The route-memory test shows that tension: its more expressive variant overfit until the audit had enough data to catch it.
- **Can the symbols come from raw input?** Everything here starts from relation words someone supplied. Deriving the vocabulary itself, from text or perception, is the larger open problem.
- **How exact must the data be?** Exact laws are brittle to noisy labels. The opt-in tolerance helps, but it trades away the guarantee. Where that trade belongs is still an open design question.

## Putting it together

A family question like "who is your daughter's grandfather?" looks trivial, and it hides everything that matters about reasoning you can trust. The chain doesn't determine the answer. The data can't show some combinations. And a question can be longer than any example. A model that derives its laws handles all three honestly: it answers with every possibility, it exposes the rules it relies on, it treats a thousand-step question exactly like a two-step one, and when a word carries two meanings it can split them from the data. It does this from 62 facts about symbols it can't read, and it lands on the same algebra an anthropologist built by hand in 1984. The open question is how far that reaches beyond families. The method is cheap enough to find out one domain at a time.

## Reproducing everything

    pip install -e .
    export CLUTRR_DIR=/path/to/clutrr/gen_train23_test2to10   # other datasets alongside it
    python -m bicyclic_kinship.benchmark    # all six datasets, about 5 seconds
    python -m bicyclic_kinship.worlds       # family trees, depth to 10,000, paths, grid
    python -m bicyclic_kinship.stress       # the stress tests, a few minutes
    pytest                                  # 59 tests

Zero dependencies, pure Python. The mathematics, the stress tests, the negative results and how the work was checked are in the technical companion, [*Deriving Composition Laws, and Trying to Break Them*](method-and-evidence.html).

**References**

*Benchmark and published systems.* Sinha, Sodhani, Dong, Pineau & Hamilton (2019), CLUTRR, EMNLP, [arXiv:1908.06177](https://arxiv.org/abs/1908.06177). Minervini et al. (2020), Learning Reasoning Strategies in End-to-End Differentiable Proving (CTP), ICML, [arXiv:2007.06477](https://arxiv.org/abs/2007.06477). Bergen, O'Donnell & Bahdanau (2021), Systematic Generalization with Edge Transformers, NeurIPS, [arXiv:2112.00578](https://arxiv.org/abs/2112.00578). Lu et al. (2022), R5, ICLR, [arXiv:2205.06454](https://arxiv.org/abs/2205.06454). Cheng et al. (2023), NCRL, ICLR, [arXiv:2303.03581](https://arxiv.org/abs/2303.03581). Khalid & Schockaert (2025), Systematic Relational Reasoning with Epistemic Graph Neural Networks, ICLR, [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/6590cb829f5ffef50050f3e5845fbb4c-Paper-Conference.pdf). Yang, Ishay & Lee (2023), Coupling Large Language Models with Logic Programming for Robust and General Reasoning from Text, Findings of ACL, [arXiv:2307.07696](https://arxiv.org/abs/2307.07696). Zhang et al. (2024), Extracting and Following Paths for Robust Relational Reasoning with Large Language Models (Path-of-Thoughts), *TMLR* 2026, [arXiv:2412.17963](https://arxiv.org/abs/2412.17963).

*Kinship algebra.* Weil (1949), appendix to Lévi-Strauss, *Les structures élémentaires de la parenté*. White (1963), *An Anatomy of Kinship*. Boyd (1969), The algebra of group kinship, *Journal of Mathematical Psychology*, [doi](https://doi.org/10.1016/0022-2496(69)90032-7). Lorrain & White (1971), Structural equivalence of individuals in social networks, *Journal of Mathematical Sociology*, [doi](https://doi.org/10.1080/0022250X.1971.9989788). Read (1984), An Algebraic Account of the American Kinship Terminology, *Current Anthropology* 25(4), [link](https://doi.org/10.1086/203160). Read (2006), Kinship Algebra Expert System (KAES), *Social Science Computer Review* 24(1), [doi](https://doi.org/10.1177/0894439305282372). Read, Fischer & Leaf (2013), What Are Kinship Terminologies, and Why Do We Care?, *Social Science Computer Review*, [doi](https://doi.org/10.1177/0894439312455914).

*Mathematics and program analysis.* Lyapin (1953), first published description of the bicyclic semigroup; Clifford & Preston (1961), *The Algebraic Theory of Semigroups*, vol. 1. Cousot & Cousot (1977), Abstract interpretation, POPL, [doi](https://doi.org/10.1145/512950.512973). Gold (1978), Complexity of automaton identification from given data, *Information and Control*, [doi](https://doi.org/10.1016/S0019-9958(78)90562-4). Ernst, Cockrell, Griswold & Notkin (1999), Dynamically discovering likely program invariants to support program evolution (Daikon), ICSE, [doi](https://doi.org/10.1145/302405.302467).

*Deriving laws and programs from data.* Angluin (1987), Learning regular sets from queries and counterexamples, *Information and Computation*, [doi](https://doi.org/10.1016/0890-5401(87)90052-6). Schmidt & Lipson (2009), Distilling free-form natural laws from experimental data, *Science*, [doi](https://doi.org/10.1126/science.1165893). Muggleton, Lin & Tamaddoni-Nezhad (2015), Meta-interpretive learning of higher-order dyadic datalog, *Machine Learning*, [doi](https://doi.org/10.1007/s10994-014-5471-y). Brunton, Proctor & Kutz (2016), Discovering governing equations from data by sparse identification of nonlinear dynamical systems, *PNAS*, [doi](https://doi.org/10.1073/pnas.1517384113). Evans & Grefenstette (2018), Learning explanatory rules from noisy data, *JAIR*, [doi](https://doi.org/10.1613/jair.5714). Chollet (2019), On the Measure of Intelligence, [arXiv:1911.01547](https://arxiv.org/abs/1911.01547). Sinha, Sodhani, Pineau & Hamilton (2020), Evaluating Logical Generalization in Graph Neural Networks (GraphLog), [arXiv:2003.06560](https://arxiv.org/abs/2003.06560). Udrescu & Tegmark (2020), AI Feynman, *Science Advances*, [doi](https://doi.org/10.1126/sciadv.aay2631). Cropper & Morel (2021), Learning programs by learning from failures, *Machine Learning*, [doi](https://doi.org/10.1007/s10994-020-05934-z). Ellis et al. (2021), DreamCoder, PLDI, [doi](https://doi.org/10.1145/3453483.3454080).


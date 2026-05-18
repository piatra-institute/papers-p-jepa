# Critique of P-JEPA

A reading of `paper/PAPER.md`, `README.md`, `AGENTS.md`, the `docs/`
collection, and the simulation source under `simulation/pjepa_sim/`. The
goal is to separate what the project genuinely demonstrates from what
its vocabulary suggests it demonstrates, and to point at the load-bearing
weaknesses a reviewer should press on.

## 1. What the project actually is

Stripped of its mathematical decoration, P-JEPA is:

- A 4-regime, 4-action, 3-probe toy world (`core/dishworld.py`) with
  hand-specified Bernoulli success/unsafe tables and probe likelihoods.
- An exact Bayesian value-of-information policy over that world, framed
  as "the P-JEPA stack."
- A scripted reach controller wrapped around Meta-World `reach-v3` with
  a hidden categorical regime applied to the action channel.
- A suite of small NumPy MLP experiments that re-derive the same 4
  hidden regimes from sampled outcomes of the hand-specified tables.
- A manifest validator and a finite-state contract checker.
- A ~30-page paper that interprets the above through sheaf theory,
  cellular cohomology, viability kernels, predictive state
  representations, active inference, and category-theoretic
  compositionality.

The engineering around the toy is unusually disciplined: a claim
ledger, an evidence matrix, a verifier per claim, an evidence-level
guard that prevents protocol checks from being counted as performance
results, gitignored generated artifacts, and a build script. That
discipline is the strongest thing about the project. The scientific
content is much thinner than the framing makes it look.

## 2. Mathematical framing vs. operational content

### 2.1 The sheaf is decorative

The paper introduces affordance presheaves, restriction maps,
coboundaries, $H^0$, $H^1$, and Hodge Laplacians on cellular sheaves.
In the implementation, the entire sheaf apparatus collapses to one
function:

```python
def obstruction(posterior):
    preds = prediction_matrix()           # 4 regimes x 4 actions
    mean = posterior @ preds
    diffs = preds - mean
    return float(np.sum(posterior[:, None] * diffs * diffs))
```

This is the posterior-weighted variance of the per-regime
success-prediction vectors. It is a perfectly reasonable scalar; it is
not cohomology, it is not a coboundary, and it does not exercise any
nontrivial sheaf structure (no cover, no overlap, no restriction map,
no nerve). The paper effectively names $\mathrm{Var}_p[\mu_a(R)]$ "the
coboundary $d\sigma$" and then calls the resulting policy a "sheaf
policy." The gluing benchmark
(`representation/gluing.py`) is the only place a literal restriction
map appears, and it is a learned linear map between two engineered
vector spaces — i.e., a least-squares alignment problem with a
sheaf-theoretic label.

A reader who has not opened the code will assume nontrivial $H^1$
classes are being computed. None are. The paper even admits this
("The results also do not isolate a uniquely sheaf-theoretic
advantage") but the title, abstract, and entire framing nonetheless
sell the sheaf as the central object.

**Recommendation.** Either commit to the math — implement an actual
cellular sheaf with a real nerve, compute $\dim H^1$ on an overlap
graph, and show that policies that minimise $H^1$ beat policies that
minimise scalar disagreement — or rewrite the paper as "active
predictive-state control with viability-aware value of information."
The latter is what the code does, and it is enough.

### 2.2 Obstruction uses success only, not unsafe

`obstruction()` is built from `prediction_matrix()` (success), not
`unsafe_matrix()`. Yet the headline claim is about *safety*. The
viability-aware behaviour comes from the *score* function
$S = \text{success} - 2 \cdot \text{unsafe} - \lambda_p \cdot
\text{probes}$ used inside the policy, not from the obstruction
signal. So the sheaf-style disagreement metric is not what carries
safety improvements; the VOI computation over a hand-written cost
function is. This further weakens the framing in §2.1.

### 2.3 Self-acknowledged collapse of the comparison

In the exact suite, `active_psr_probe` (a plain exact value-of-
information policy) and `p_jepa_stack` produce identical results
(`0.853 / 0.081 / 1.987 / 0.165 / 0.651`). The paper acknowledges
this. It is fatal to the narrow claim that *sheaf-style obstruction*
is the source of the gain. The gain comes from doing exact Bayesian
VOI in a 4-state world — which has been a textbook exercise since at
least the 1960s. The "stack" adds an obstruction-based coherence gate
on top, but in this world the gate is redundant.

## 3. The simulation is a hand-coded specification, not learning

`ACTION_MODEL` and `PROBE_LIKELIHOOD` are dictionaries the author
typed in. The "exact" evaluator enumerates over the 3 probes × 2
outcomes × 4 actions tree against those same tables. The "P-JEPA
stack" then chooses actions by computing
$\arg\max_a \mathbb{E}_p[S(a, R)]$ over the same tables. Every result
in §9 of the paper is a deterministic function of the tables and the
policy expression. There is no learning anywhere in the §9 numbers,
and no source of variance other than choice of policy expression.

The neural experiments
(`representation/{neural, neural_active, learning}.py`) train tiny
MLPs to map sampled outcomes back to the same 4 hidden regimes those
samples were generated from. Reaching purity $1.000$ in such a
setting is not surprising; it is the expected behaviour of any
reasonable clustering algorithm operating on samples from 4
linearly separable Bernoulli sources. Calling this a "neural
P-representation" is rhetorically generous.

The honest version of the table in §9 is:

> Under exact Bayesian inference with a hand-written cost function,
> exact value of information beats acting from the prior on a 4-state
> bandit. Engineered fingerprints recover the 4 states.

That is true. It is not, on its own, evidence for a new
representation-learning principle. It is evidence that Bayes works on
problems small enough to enumerate.

## 4. The Meta-World adapter

§10 is the only place the project leaves its own toy world, and it
does so in a thin way:

- The controller is **scripted**, not learned. The paper says so.
- The "hidden regime" is a categorical scalar (`nominal / slippery /
  fragile / heavy`) that scales action magnitude, adds Gaussian noise,
  and changes an unsafe-action threshold. It is functionally a
  parameter sample, not a regime in any contact-mechanics sense.
- The "probes" are wrapper-defined synthetic measurements with
  author-chosen likelihoods. They are not physical probes the agent
  has discovered.
- The 100-episode budget yields differences such as "score 0.860 vs
  0.766" with no reported variance, confidence interval, or
  multi-seed dispersion.

The result is consistent with the toy world story: exact VOI over a
small categorical hidden variable selects useful probes. It does not
constitute evidence about Meta-World, robotics, or any continuous
control problem in a meaningful sense.

A reviewer of a robotics venue should treat §10 as a sanity check
that the API around the toy still works once you call
`env.reset()` on `reach-v3`. It is not a Meta-World result.

## 5. The KTH experiment is correctly negative — but the framing buries it

The paper deserves credit for running the KTH sample-video benchmark
and reporting that static appearance (`0.896`) beats both passive
next-frame (`0.805`) and temporal motion (`0.623`). This contradicts
the rendered-video surrogate in §9, which the paper uses to argue
that passive video prediction is the wrong objective.

The honest reading is: the only non-rendered video test in the
project failed in the direction opposite to the paper's main story.
The paper treats this as a "smoke test" and a "diagnostic," and
reframes it as motivation for a "next validity test" rather than as
counter-evidence. Both readings are defensible, but the asymmetry is
worth flagging: synthetic rendered video that supports the story is
reported as evidence; real video that contradicts the story is
reported as a sample-size limitation.

The fact that the only honest video result currently in the
repository is *against* the story is barely visible in the abstract.

## 6. Citation hygiene

Several references are forward-dated to **2026** or later:

- Mur-Labadia et al. (2026), "V-JEPA 2.1," arXiv:2603.14482 — the
  arXiv ID format is `YYMM.NNNNN`; `2603` is not a valid month.
- Logical Intelligence (2026a, 2026b), Kona and Aleph product pages.
- Assran et al. (2025), V-JEPA 2 — plausible but should be verified.

The document is dated May 2026. Either the project is being written
to a future submission target and is consciously citing pre-prints
that do not exist yet, or the references are placeholders that were
never updated. Either way, a reviewer in 2026 will check these and
the malformed arXiv ID will be the first thing they notice.

Cross-PIATRA citations are forbidden by `AGENTS.md`, which is a good
norm, but the same norm should apply to citing your own forward-dated
work-in-progress.

## 7. Evidence inflation

The paper makes many claims of the form "X benchmark passes" without
distinguishing:

- Mechanism claims that are *definitionally true* under the hand-coded
  tables (most of §9, all the "purity 1.000" numbers).
- Reasonable empirical claims with single seeds and tiny n
  (the Meta-World runs, 100 episodes, no CI).
- Protocol claims that prove infrastructure refuses bad data, not
  that any actual data has been processed (manifest validators).

The `docs/SCIENTIFIC_CLAIMS.md` and `docs/CLAIM_LEDGER.md` documents
do separate these — and the `evidence_claims.py` guard is one of the
best ideas in the project — but the paper itself flattens them into a
long sequence of numeric assertions. A typical paragraph in the
abstract reads like a results section; an unfamiliar reader will
struggle to know which of the ~30 quoted numbers correspond to
empirical findings and which are tautologies over hand-typed tables.

The five-seed "robustness sweep" is described as not being a
"statistical confidence interval"; it should also be described as not
being statistics at all. Five deterministic seeds with no
preregistered effect size estimate cannot distinguish a true effect
from noise at any defensible alpha. Reporting "mean advantage over
entropy is only $0.005$ and one seed favours entropy" is candid, but
the right conclusion is that **no claim of superiority over entropy
probing is supported by this study** — not "slightly better on
average."

## 8. Threshold engineering

`docs/ACTION_GROUNDING_CHALLENGE.md` lists thresholds (`> 0.30`,
`> 0.25`, `> 0.20`, `> 0.18`, etc.) and the table immediately below
shows the current observed values (`0.520`, `0.520`, `0.213`,
`0.224`, `0.390`). Every threshold is met with substantial margin
*and* every threshold appears to have been set to be met by the
current numbers. There is no record of a threshold being chosen
before the experiment was run, and no documented protocol for
adjusting it.

This is the dressing of preregistration without its substance. A
challenge whose pass/fail bar is read off the current run is a
report of the current run, not a test. If these thresholds are meant
to be a contract for future versions, they should be tagged in a way
that makes adjustments visible (e.g., a version-controlled `expected
margin = 0.18; pass margin = 0.10` separation, with an explicit log
when margins drop).

## 9. Code quality observations

The implementation is, on balance, careful:

- Deterministic execution with seeded RNGs; no nondeterministic GPU
  paths.
- Per-claim verifier scripts, one JSON artifact per benchmark, an
  audit-summary writer that catalogues every verifier.
- An evidence-level registry that refuses to count protocol checks as
  performance claims.
- A clear separation between `core/`, `benchmark/`, `representation/`,
  `perception/`, `external/`, `formal/`, and `verification/`.

Two structural concerns:

1. **Volume of CLI entrypoints.** `simulation/pjepa_sim/cli/` has
   ~20 separate commands. `verify_all` is the right unified entry,
   but the per-CLI Markdown reports and JSON files duplicate
   information that the audit already collects. A reviewer trying to
   reproduce a single number has to know which of the ~20 commands
   regenerates it; the claims summary partly addresses this but the
   coupling is fragile.

2. **The "neural" code is engineered to succeed.** `TinyMLP`
   (representation/neural.py:73) uses tanh hidden, sigmoid output,
   Adam, 900 epochs, and a hand-tuned learning rate of `0.025`. It
   maps 4-dim sensors plus test identity to a 2-vector. The hidden
   layer width is fixed by the calling code. The training set is
   synthesised from `ACTION_MODEL`. Calling this configuration a
   "neural P-representation" and then reporting that it "matches the
   engineered reference" overstates what is happening: a 2-layer MLP
   fits 4 Bernoulli distributions. The interesting test would be
   identical training data passed through a *frozen* random encoder;
   the gap would likely be small.

## 10. What the project does support

To be fair to the work, here is the strongest defensible reading:

- A clear *mathematical framework* for what a praxis-grounded
  representation would need to do — predictive sufficiency under
  intervention, viability, local-to-global consistency, skill
  composition. This part of the paper (§2–§7) is a useful synthesis
  of PSR, bisimulation metrics, viability theory, active perception,
  and applied sheaf theory. It would stand on its own as a position
  paper.
- A *negative result* against pure passive next-frame prediction
  under controlled visual style shift. The mechanism is small but the
  point is real: if your evaluator shifts visual appearance, your
  passive features will not transfer.
- A clean *probe-selection mechanism* in a finite hidden-regime world,
  with the right comparison against entropy probing, random probing,
  and an oracle. The fact that exact VOI matches the "stack" tells
  you the mechanism is correctly Bayesian on that world.
- *Infrastructure* — manifest validators, evidence-level guards,
  finite-contract export, claim ledger — that is more careful than
  most ML research codebases. The discipline is reusable independent
  of the scientific claims.

## 11. Recommendations for the next pass

In rough order of payoff:

1. **Cut the abstract to a third of its length.** The current abstract
   is a dense list of numeric results that no reader can verify
   without opening the repo. Replace it with the contribution
   ("intervention-sufficiency criterion, sheaf-style coherence
   diagnostic for active probing, executable hidden-regime
   benchmark") and a single concrete result.
2. **Demote or remove the sheaf framing**, or implement a real one.
   The current code computes a posterior-weighted variance; the paper
   calls it cohomology. Either is fine; both is not.
3. **Run *one* real benchmark.** Milestone 1 in
   `docs/NEXT_VALIDITY_TESTS.md` (full KTH or Something-Something V2)
   is the right priority. Until then, every "video representation"
   claim is a rendered-pixel surrogate.
4. **Drop the forward-dated citations.** Mur-Labadia et al. (2026)
   with arXiv ID `2603.14482` is the kind of detail that ends a
   review. Either cite real papers or remove the references.
5. **Separate mechanism claims from empirical claims in the paper.**
   The `EVIDENCE_MATRIX.md` separation belongs in the paper text,
   not only in the repo.
6. **Stop calling 4-regime Bernoulli fits "neural."** The MLPs are
   fine as fitters; they do not carry the implications the word
   "neural" carries in 2025-26 ML.
7. **Replace deterministic seed sweeps with proper statistics.** With
   100 Meta-World episodes per condition, a paired bootstrap over
   episodes is straightforward and would give real confidence
   intervals.
8. **Either run Kona/Aleph or remove §11.** Exporting a contract
   format that no external verifier consumes is plumbing, not a
   verification result. The current honesty in saying so is good,
   but the section then has nothing to add.

## 12. Bottom line

P-JEPA is a careful piece of *engineering* around a *thin* scientific
finding wrapped in *heavy* mathematical decoration. The engineering
discipline — claim ledger, evidence levels, verifier-per-claim,
generated-artifact policy — is unusually good and worth imitating.
The mathematical framing oversells what the code does. The empirical
content is exact Bayesian VOI on a hand-specified 4-state world plus
a scripted Meta-World wrapper, and the one honest non-rendered video
test points the other way.

A trimmed, less ambitious version of this paper — "active probing for
hidden-regime manipulation under viability constraints, with a
mechanism benchmark and an honest negative video result" — would be a
solid workshop paper. The current paper claims to be the foundation
for a new joint-embedding family. It is not yet.

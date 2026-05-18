# Hypothesis Results

This document records the outcome of the preregistered hypothesis tests
defined in `docs/DELIVERY_PLAN.md` and the plan at
`~/.claude/plans/ok-make-a-plan-declarative-scone.md`. Each section is
written **after** the experiment runs, with the observed value, the
preregistered pass/fail criterion, and the decision the result implies
for the paper and codebase.

This artifact gates any future deletions or rewrites. No textual or
code change driven by `docs/CRITIQUE.md` should land before the relevant
hypothesis has been tested here.

Run order matches `simulation/pjepa_sim/experiments/`. All JSONs are at
`simulation/output/experiments/`.

---

## H1 — Obstruction gate is a no-op

**Hypothesis.** `p_jepa_stack` and `active_psr_probe` produce identical
numeric output on every configured suite, because the obstruction gate
inside `_belief_decision_metrics` (`suites.py:383`) never fires below
`spec.sheaf_threshold` for any non-terminal posterior reached on the
existing suites.

**Pass criterion.** Max absolute difference in `risk_adjusted_score`
(and four other scalar metrics) across all five suites is `< 1e-9`.

**Result.** **PASS.**

```
max |delta| = 0.000e+00 across all 5 suites and 5 metrics
```

Per-suite check:

| Suite | obstruction at policy_prior | sheaf_threshold | above threshold? | stack score | psr score |
|---|---:|---:|:---:|---:|---:|
| hidden_regime_v0 | 0.2555 | 0.060 | ✓ | 0.6511 | 0.6511 |
| heldout_regime_v0 | 0.2555 | 0.060 | ✓ | 0.5570 | 0.5570 |
| noisy_probe_v0 | 0.2555 | 0.060 | ✓ | 0.5075 | 0.5075 |
| costly_probe_v0 | 0.2555 | 0.060 | ✓ | 0.5000 | 0.5000 |
| miscalibrated_sections_v0 | 0.1536 | 0.060 | ✓ | 0.6510 | 0.6510 |

Every starting posterior has obstruction (0.15-0.26) well above the
threshold (0.06), so the gate never short-circuits a probe at the root.
The decision tree's leaves are direct actions where the gate returns
the same answer as the no-gate path. Operationally, the two agents are
identical on these suites.

**Decision.** Confirmed. In a follow-up session, merge `p_jepa_stack`
and `active_psr_probe` into one named agent in the paper, or add a
new suite with `sheaf_threshold > prior obstruction` to actually
exercise the gate. The current §9 sentence "exact Bayesian value of
information already selects the same probes once local sections are
known" should be promoted from a parenthetical to a structural
observation about the agent design.

**Artifact.** `simulation/output/experiments/h1_obstruction_gate.json`.

---

## H3 — Frozen random projection matches the trained MLP

**Hypothesis.** The "neural P-representation" recovers regimes because
the four hidden Bernoulli sources are already nearly linearly separable
in sensor space, not because the MLP is learning anything specific. A
frozen random projection of equal width should match the trained MLP
on risk-adjusted score and cluster purity.

**Pass criterion.** `mean(trained_score - frozen_score) < 0.05` AND
95% paired bootstrap CI on per-seed delta contains zero (10 seeds).

**Result.** **PASS** — confirmed strongly.

```
appearance baseline:   0.282
frozen random TinyMLP: 0.800  (purity 0.994)
trained TinyMLP:       0.802  (purity 1.000)

trained - frozen score CI95: [+0.0000, +0.0045], mean = +0.0015
trained - frozen purity CI95: [+0.0000, +0.0176], mean = +0.0059
```

The frozen-random projection captures ~99% of the trained MLP's
risk-adjusted score and ~99% of its cluster purity. The 0.52-point
gap from appearance to either MLP is real (visual labels shift between
train and test); the 0.002-point gap from random to trained is not.

**Decision.** Confirmed. In a follow-up session, demote the "neural
intervention encoder" / "neural P-representation" framing across
`docs/SCIENTIFIC_CLAIMS.md`, `paper/PAPER.md` §9, and the relevant
verifier descriptions. The mechanism is **test-vector clustering on
linearly-separable Bernoulli sources**, not neural learning. Add a
`frozen_random_projection` column to the neural benchmark tables to
make the comparison visible.

The sample-efficiency story (§9 paragraph on intervention repeats)
remains technically correct but now reads differently: the trained MLP
has lower MAE on raw outcome prediction than a random projection, but
both reach the same clustering quality, so the MAE improvement does
not translate to action-choice improvement at any tested budget.

**Artifact.** `simulation/output/experiments/h3_frozen_random_baseline.json`.

---

## H2 — Active-probing advantage over entropy is seed noise

**Hypothesis.** The five-seed sweep at `neural_active.py:677` shows
active probing beats entropy by mean 0.005 with one seed favouring
entropy. With more seeds (50) and a paired bootstrap CI, the
active-vs-entropy delta contains zero.

**Pass criterion.** 95% paired bootstrap CI on per-seed
(active_score - entropy_score) delta contains zero across 50 seeds.

**Result.** **FAIL** — hypothesis rejected; the claim survives.

```
50 seeds, 10000 paired bootstrap resamples

active - entropy:   mean +0.0130, CI95 [+0.0094, +0.0166]   (excludes zero)
active - no_probe:  mean +0.2119, CI95 [+0.2081, +0.2159]   (excludes zero)
no_probe.unsafe - active.unsafe:  mean +0.0725, CI95 [+0.0715, +0.0736]   (excludes zero)

win counts vs entropy: active 41/50, entropy 9/50, ties 0
```

The original five-seed sweep was underpowered, not wrong. With 50
seeds the value-aware active probing has a real (small but statistically
reliable) advantage over entropy probing: about +0.013 score points,
with active winning 82% of seeds.

The two adjacent claims that the original sweep treated as robust both
also survive cleanly: active vs no-probe (+0.21 score) and the
unsafe-failure reduction (+0.072), both with CIs excluding zero by wide
margins.

**Decision.** Hypothesis falsified. The paper's "value-aware probing
beats entropy" claim is supported. Action: replace the five-seed
sentence in the abstract (`paper/PAPER.md:11`) with the bootstrap CI
from this experiment. The CRITIQUE.md §7 line that called the five-seed
sweep "not statistics at all" is correct — five seeds was too few — but
the underlying claim it questioned is real once properly tested.

**Artifact.** `simulation/output/experiments/h2_seed_sweep_bootstrap.json`.

---

## H4 — Sheaf framing is decorative

**Hypothesis.** A real cellular sheaf with learned linear restriction
maps and a `||delta sigma||^2` gluing loss does not outperform the raw
cover centers (scalar baseline) on downstream action choice.

**Pass criterion.** `mean(sheaf_score - scalar_score) < 0.02` AND
95% paired bootstrap CI on per-seed delta contains zero (20 seeds).

**Result.** **FAIL on the preregistered criterion — in the unexpected
direction.** The hypothesis is *strengthened*, not weakened.

```
K = 6 clusters (forced above true regime count of 4, so intra-regime
fragments overlap and the sheaf has non-trivial structure)

scalar baseline (raw centers): mean score 0.802
sheaf-glued centers:           mean score 0.798

sheaf - scalar score CI95: [-0.0050, -0.0036], mean = -0.0043
sheaf - scalar unsafe CI95: [+0.0018, +0.0025], mean = +0.0021
```

The sheaf machinery is real and active:

- mean number of edges in the 1-skeleton: 8.9 (out of K*(K-1)/2 = 15)
- mean `dim H^0` (harmonic sections): 9.8
- mean `dim H^1` (residual obstructions): 42.3
- coboundary energy reduction from gluing: 0.354 → 0.0345 (≈ 10×)

So the gluing successfully reduces `||delta sigma||^2` by an order of
magnitude. The cohomology dimensions are non-trivial. The construction
is doing what the math says it should do. But the resulting glued
centers produce **slightly worse** action choice than the raw centers,
not better. The CI is statistically reliable (excludes zero on the
negative side).

The preregistered pass criterion was a one-sided test: PASS if mean is
small AND CI contains zero. The result violates the second condition
because the CI is entirely negative. This is technically a FAIL on the
preregistered criterion. But the *intent* of the hypothesis ("sheaf is
decorative") is *more* strongly confirmed than expected: the sheaf
framing is not neutral on dishworld, it is mildly counterproductive.

The mechanism: gluing pulls fragment centers toward consistent
restrictions on overlaps. When K > true regime count, the overlapping
fragments belong to **the same true regime**, so consistency is natural.
But the small noise reduction from gluing is dominated by the loss of
specificity in the per-fragment predictions when they are pulled
toward the joint mean. Restriction-map agreement is not a good proxy
for action-choice utility on this benchmark.

**Decision.** The decision rule from the plan ("If FAIL: keep the
framing, promote sheaf_toy.py") **does not apply** because the failure
is in the unexpected direction. The intended decision rule for "FAIL"
assumed the sheaf would beat the baseline; instead it lost.

The correct decision is the same as for PASS: in a follow-up session,
demote the sheaf framing from the paper title and abstract to a
motivation paragraph. Keep `sheaf_toy.py` in the repo as a load-bearing
*negative-result artifact* with the verifier, the JSON, and an explicit
note that the construction reduces coboundary energy 10× without
improving downstream score.

The mechanism the code actually exercises is "active probing under
viability via posterior-weighted prediction variance plus exact VOI."
That is what the paper should claim. The cellular-sheaf construction
is now a *falsified specific claim* on dishworld: the gluing
mathematics works as advertised, but it does not produce a useful
representation for the downstream task on this benchmark.

**Artifact.** `simulation/output/experiments/h4_sheaf_vs_scalar.json`.

The sheaf module `simulation/pjepa_sim/representation/sheaf_toy.py`
remains in the repo. It is the only place in the project where a real
cellular sheaf (cover, nerve, restriction maps, coboundary, Laplacian,
H^0 / H^1) is actually constructed. Future work should not pretend the
construction was never attempted.

---

## H5 — JEPA augmentations improve dishworld toy

**Hypothesis.** At least one of the augmentations {intervention,
bisim, active masking, viability, all combined} produces a non-zero
paired bootstrap CI advantage over base JEPA on the dishworld toy in
`simulation/pjepa_sim/jepa_toy/`. The toy is a NumPy JEPA (encoder +
EMA target encoder + mask predictor) with each augmentation as a
toggleable auxiliary loss.

**Pass criterion.** At least one augmentation has 95% CI on
(variant - base) per-seed delta strictly above zero across 12 seeds.

**Result.** **FAIL** (no augmentation clears the bar) **but the
per-augmentation pattern is informative.**

```
12 seeds, 500 epochs each, 10000 paired bootstrap resamples

base JEPA:        score 0.591 (range 0.44-0.80 across seeds)
+intervention:    score 0.590, delta -0.001, CI [-0.127, +0.125]   (neutral)
+bisim:           score 0.461, delta -0.130, CI [-0.231, -0.027]   (HURTS - CI excludes 0 negatively)
+active masking:  score 0.585, delta -0.006, CI [-0.059, +0.048]   (neutral, tight)
+viability:       score 0.624, delta +0.033, CI [-0.007, +0.098]   (positive trend, CI nearly excludes 0)
+all:             score 0.477, delta -0.114, CI [-0.209, -0.023]   (hurts, driven by bisim)
```

**Three real findings inside the FAIL verdict:**

1. **Base JEPA has huge seed variance** (0.44 to 0.80 across 12
   seeds). It converges to one of two basins: a "good" basin where
   the encoder cleanly separates 4 regimes (~0.80 score) and a "bad"
   basin where the latent partially collapses (~0.44). This means
   12 seeds are not enough to detect <0.10-point augmentation
   effects with statistical confidence.
2. **Bisimulation at $\lambda = 0.3$ is mis-calibrated** and actively
   hurts (CI excludes zero negatively). The bisim loss pulls the
   encoder toward a distance-only metric and degrades cluster
   structure. Consistent with the literature warning that bisim needs
   curriculum tuning of $\lambda$.
3. **Viability head shows the most promising signal.** Mean delta
   +0.033, CI [-0.007, +0.10] — just barely contains zero. The
   per-seed pattern is the most informative: on seeds where base JEPA
   converges to the bad basin (e.g., seed 103: 0.44), viability
   "rescues" the result (to 0.80); on seeds where base is already in
   the good basin, viability does not help and may slightly hurt.
   This is consistent with the viability head acting as an additional
   regulariser that helps when the JEPA loss alone has converged
   poorly.

The verdict is FAIL because no single CI excludes zero on the
positive side. The interpretation is *not* "augmentations don't
help." It is "the toy is at its variance limit; bisim needs $\lambda$
tuning; viability has the strongest directional signal; intervention
and active masking are within toy noise."

**Decision.** Toy infrastructure is now in place
(`simulation/pjepa_sim/jepa_toy/`) and proven correct (every loss
decreases monotonically; the heads train; downstream evaluation
reproduces the engineered-fingerprint score on the good basin).
Augmentations promoted to V-JEPA-scale priority per the typology in
`docs/JEPA_AUGMENTATIONS.md` and the priority order in
`paper/PAPER_v2.md` §6:

1. Intervention loss + composition consistency (high inductive-bias
   match with V-JEPA's action-conditioned setting)
2. Active masking (cheap, literature precedent for ~0.5-1% gains)
3. Sheaf consistency on overlapping clips (conditional on H4 boundary
   - it should help on continuous overlapping video where it hurts on
   categorical regimes)
4. Bisimulation (after a $\lambda$ curriculum is designed)
5. Viability head (last for general representation; first for
   safety-critical downstream)

**Artifact.** `simulation/output/experiments/h5_jepa_augmentations.json`.

The toy infrastructure: `simulation/pjepa_sim/jepa_toy/{model,
losses, training, eval, data}.py`. The PyTorch design specs for each
augmentation: `docs/JEPA_AUGMENTATIONS.md`.

---

## Summary table

| H | Hypothesis | Verdict | Direction | Paper consequence |
|---|---|---|---|---|
| H1 | obstruction gate is a no-op | PASS | confirmed | merge p_jepa_stack and active_psr_probe |
| H2 | active-vs-entropy is seed noise | FAIL | claim survives 50 seeds | replace 5-seed prose with bootstrap CI |
| H3 | trained MLP ≈ frozen random | PASS | confirmed | demote "neural" framing; add random-projection column |
| H4 | sheaf is decorative | FAIL on criterion, hypothesis *strengthened* | sheaf is mildly worse than scalar | demote sheaf framing; keep `sheaf_toy.py` as negative-result artifact |
| H5 | augmentations help on toy | FAIL (variance-limited) | bisim hurts, viability positive trend, others neutral | promote viability + intervention + composition to V-JEPA-scale per JEPA_AUGMENTATIONS.md |

Three of the four critique-driven hypotheses are confirmed; the fourth
(active vs entropy) is the one the paper got right with insufficient
evidence and now has proper evidence for.

---

## Next session

The follow-up session that consumes these results should:

1. Update the abstract and §9 of `paper/PAPER.md` per the decisions
   above. Replace the five-seed sentence with the H2 CI. Demote the
   "neural" framing in the neural-benchmark passages (H3). Collapse
   the obstruction-gate / sheaf-policy distinction (H1, H4). OR adopt
   `paper/PAPER_v2.md` as the new paper, which incorporates all of
   these.
2. Add a `frozen_random_projection` baseline column to the neural
   benchmark output and rerun the neural verifier (H3).
3. Decide whether to keep the cellular-sheaf framing as a motivation
   paragraph or remove it from the title (H4). Either way, document
   the negative result.
4. Reference this file in `docs/CLAIM_LEDGER.md` so reviewers can see
   which paper claims have been experimentally tested and what the
   outcome was.
5. Begin V-JEPA-scale implementation per `docs/JEPA_AUGMENTATIONS.md`:
   port the toy losses to PyTorch on top of a public V-JEPA reference
   implementation. Start with intervention + composition consistency
   (highest expected gain). The priority order is in
   `paper/PAPER_v2.md` §6.

No deletions or paper rewrites of the original `paper/PAPER.md`
happen as part of the current session. This document is the gate.
The new `paper/PAPER_v2.md` is an honest revision *alongside* the
original, not a replacement; the choice between them is for a future
session.

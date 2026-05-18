# Delivery Plan

A concrete plan to make P-JEPA do what the paper claims it does. This
document supersedes `NEXT_VALIDITY_TESTS.md` as the working plan; that
document remains valid as a list of validity tests, but it does not
order the work or force a decision about what the paper is actually
about.

The plan is structured as: (0) the pivot decision, (1–3) three
candidate deliveries with concrete technical contents, (4) what to
delete regardless of which path is chosen, (5) a sequenced milestone
plan, and (6) the honest minimum if only one quarter is available.

## 0. The pivot

The current paper tries to be three things at once and is none of
them. Pick one before any further work.

| Path | Paper identity | Core math | Engineering cost | Defensibility |
|---|---|---|---|---|
| A | Active probing under viability for hidden-parameter MDPs | Bayesian VOI + control barrier functions | Months, one person | High |
| B | Sheaf-theoretic learning architecture | Cellular sheaf cohomology, end-to-end differentiable | 6–12 months | Medium, novel |
| C | JEPA variant for embodied video | Masked predictive learning, EMA target | 6–18 months, GPUs | Medium, crowded |

These are different papers. Path A is the smallest credible upgrade
and should be done first regardless. Path B is the only path that
earns the current title. Path C is the only path that earns the
"JEPA" suffix.

The remainder of this document specifies the technical contents of
each path.

## 1. Path A — Active probing for hidden-parameter MDPs

### 1.1 Scope

Drop the sheaf framing entirely. Position the paper as: under
parameter uncertainty in a continuous-control task with viability
constraints, value-of-information probing improves safety relative to
no probing and entropy probing, *when probes carry information and
are cheaper than the unsafe events they prevent*.

This is what the current code actually demonstrates in a 4-state
world. Path A puts the same mechanism into an environment where the
result would be non-trivial.

### 1.2 Required changes

1. **Environment.** Replace `dishworld` and the scripted Meta-World
   wrapper with MuJoCo manipulation under domain randomisation.
   Candidates: `panda-gym` (pick-and-place, push, slide),
   `ManiSkill2` (pick-cube, peg-insertion-side, plug-charger),
   `robosuite` (door-opening, nut-assembly). Hidden parameter vector
   $\theta \in \mathbb{R}^d$ samples per episode and includes at
   minimum: friction coefficient, object mass, gripper force limit,
   joint compliance. $\theta$ is never observed.
2. **Observations.** Pixels (or pixels + proprioception). No
   hand-typed sensor vectors. The encoder must do real work.
3. **Action space.** Continuous joint or end-effector control. A
   subset of low-energy actions (small displacements, light contacts,
   short reaches) function as candidate probes. The agent learns
   which candidate actions are informative; nothing is labelled as
   a probe in advance.
4. **Safety metric.** Either:
   - **Control barrier function:** define $b(q) \geq 0$ over the
     state (object inside workspace, contact force below limit, end
     effector outside collision zone). Unsafe rate = fraction of
     rollouts that violate $b$.
   - **HJ reachability:** compute (or approximate via a learned
     value function) the backward reachable tube of an unsafe set
     under the worst-case disturbance, report probability of
     entering it.
   The Bernoulli `unsafe` flag drawn from a typed table must be gone.
5. **Posterior over latents.** A small recurrent or attention-based
   encoder produces a posterior $q_\phi(\theta \mid h_t)$ over the
   hidden parameter vector. Trained by maximum likelihood on
   transition-prediction loss, *not* by handing it the true $\theta$.
6. **Probe selection.** Two policies to compare:
   - Expected entropy reduction under $q_\phi$.
   - Expected $\Delta$score $= \mathbb{E}[V(s_{t+k}) - V(s_t)] -
     c(\alpha)$ subject to $\Pr(\text{unsafe})\leq\delta$,
     evaluated by Monte Carlo rollout in a learned dynamics model.
7. **Baselines.** Behavioural cloning. Random probing (same budget).
   Entropy probing. No-probe policy. Oracle that observes $\theta$.
   All under matched compute.
8. **Statistics.** At least 200 episodes per condition. Paired
   bootstrap confidence intervals over episodes for every reported
   delta. No deterministic seed sweeps as evidence.

### 1.3 Acceptance criteria

- Value-of-information probing improves risk-adjusted score over
  no-probe and entropy baselines with non-overlapping 95% bootstrap
  CIs on at least one task.
- The advantage shrinks as predicted by theory when (i) the encoder
  is told $\theta$ (no value to probe), (ii) probes are made
  uninformative (no $\Delta$entropy), (iii) probe cost is raised
  above unsafe-event cost (negative VOI).
- The encoder's posterior is well-calibrated (reliability curve, ECE
  $< 0.05$ on held-out trajectories).
- A held-out parameter region (sample $\theta$ outside the training
  distribution) shows the same qualitative ordering.

### 1.4 Estimated effort

Three months, one researcher with MuJoCo familiarity and a single
A6000-class GPU.

## 2. Path B — Sheaf-theoretic learning architecture

This is the only path that earns the title of the current paper.
It is also the most expensive. Do Path A first.

### 2.1 What "sheaf" has to mean operationally

Not: scalar posterior-weighted variance of a $4 \times 4$ table.

Yes: a cellular sheaf $\mathcal{F}$ over a learned 1-complex (the
nerve of a learned cover), with vector-space stalks, learned linear
restriction maps, a differentiable coboundary $\delta$, and a
reported $\dim H^0$ and $\dim H^1$ that change during training.

### 2.2 Concrete construction

1. **Learned cover.** $K$ local experts $\{f_i\}_{i=1}^K$. Each
   carries a soft gate $g_i(x) \in [0,1]$ over observations,
   parameterised by a small network. Region of validity
   $U_i = \{x : g_i(x) > \tau\}$. Overlap support
   $U_i \cap U_j = \{x : g_i(x) > \tau \wedge g_j(x) > \tau\}$.
   $K$ can be fixed or grown by splitting on persistent prediction
   error (the current online cover code is a starting point).
2. **Vertex stalks.** $\mathcal{F}(i) = \mathbb{R}^d$ — the local
   expert's prediction on a shared test bank, or its internal
   parameter projection. Dimension chosen by the design.
3. **Edge stalks.** For each non-empty overlap $(i,j)$,
   $\mathcal{F}(ij) = \mathbb{R}^{d'}$, learned. $d'$ may be smaller
   than $d$ (restriction can lose information).
4. **Restriction maps.** $\rho_{i,ij} \in \mathbb{R}^{d' \times d}$
   learned per overlap. Real parameters. Symmetric construction for
   $\rho_{j,ij}$.
5. **Coboundary.** On overlap samples $x \in U_i \cap U_j$:
   $$(\delta\sigma)(ij; x) = \rho_{i,ij}\,\sigma_i(x) -
   \rho_{j,ij}\,\sigma_j(x).$$
   Training loss includes $\mathcal{L}_{\mathrm{glue}} =
   \sum_{ij} \mathbb{E}_{x \in U_i \cap U_j}
   \|(\delta\sigma)(ij; x)\|^2$.
6. **Sheaf Laplacian.** Assemble the global coboundary operator
   $\delta_0$ from the per-overlap restrictions. The sheaf Laplacian
   $L_0 = \delta_0^\top \delta_0$ acts on $\bigoplus_i \mathcal{F}(i)$.
   $\dim H^0 = \dim \ker L_0$. On the 1-skeleton,
   $\dim H^1 = \dim C^1 - \mathrm{rank}\,\delta_0$.
7. **Reporting.** Log $\dim H^0$, $\dim H^1$, and
   $\|\delta\sigma\|^2$ throughout training. Show that
   $\|\delta\sigma\|^2$ decreases with the gluing loss enabled, that
   $\dim H^1$ can be driven to zero on tasks with coherent global
   sections, and that it stays positive on tasks with structurally
   incompatible regimes.

### 2.3 The decisive ablation

Two models, identical except for the sheaf:

- **Sheaf model.** Above construction, with learned restriction maps
  and $\|\delta\sigma\|^2$ in the loss.
- **Scalar model.** Replace $\delta$ with the identity, replace the
  gluing loss with posterior-weighted variance of vertex predictions
  (what the current code computes). Same parameter count, same
  training data, same compute.

If the sheaf model does not outperform the scalar model on
downstream control under hidden-regime parameter shift, the sheaf
framing is not earned. Report the result either way. A clean
negative would be a publishable contribution on its own.

### 2.4 Acceptance criteria

- The sheaf model beats the scalar ablation on risk-adjusted score
  under held-out parameter shift, with non-overlapping bootstrap CIs.
- $\dim H^1$ is non-trivially affected by training (not constant,
  not always zero, not always full rank).
- At least one constructed task has provable structural $H^1 \neq 0$
  (e.g., a Möbius-style sign-flip overlap that no choice of $\sigma$
  can resolve), and the agent's reported $H^1$ matches.
- The benchmark is run in the Path A environment, not in
  `dishworld`.

### 2.5 Estimated effort

Six months on top of Path A. Requires someone comfortable with both
PyTorch autograd and applied sheaf theory (Hansen, Ghrist, Bodnar,
Curry as starting references; the existing `gluing.py` is a
sketch, not a starting point).

## 3. Path C — JEPA variant for embodied video

The only path that earns the "JEPA" suffix.

### 3.1 Required implementation

1. **PyTorch.** Not NumPy. The current `TinyMLP` cannot be scaled.
2. **Encoder.** ViT-B or ViT-L for images, a tube-ViT or
   factorised space-time encoder for video. Initialised from
   I-JEPA/V-JEPA weights for a fair starting point, or trained from
   scratch if the comparison is end-to-end.
3. **Target branch.** EMA-updated target encoder with the standard
   schedule (momentum $0.996 \to 1.0$ over training).
4. **Predictor.** Conditioned on an *action-test embedding*
   $e(\alpha)$ instead of (or in addition to) a mask location. The
   action test is either a real action vector from a robot dataset
   or a controlled augmentation (camera motion, simulated
   intervention) in pretraining.
5. **Losses.** Standard JEPA prediction loss, plus the Path B
   gluing loss if running Paths B and C together, plus an
   intervention loss $\mathcal{L}_{do}$ that requires action-effect
   prediction to match observed post-intervention frames.
6. **Data.** SSv2 (220K clips), Ego4D-derived clips (subsample to
   a manageable scale), or DROID (76K episodes with action
   metadata). Not rendered $12 \times 12$ pixels.
7. **Evaluation.** Linear probe and fine-tuning on at least one
   action-recognition benchmark (SSv2) and one intervention task
   (a robot manipulation benchmark from the manifest path). Compute
   matched against I-JEPA-B/L and V-JEPA-B/L baselines.

### 3.2 Acceptance criteria

- P-JEPA does not collapse standard video accuracy versus V-JEPA at
  matched parameters and FLOPs.
- P-JEPA improves on at least one intervention-relevant downstream
  metric (action prediction, regime classification on held-out
  parameter shift, success rate of a frozen-feature policy head)
  with non-overlapping CIs.
- Negative result reported honestly if it occurs.

### 3.3 Estimated effort

Twelve to eighteen months, one researcher plus GPU access at the
scale of at least 4×H100 for a meaningful pretraining run, or
2–3×A100-months if subsampling aggressively. Cannot be done in
NumPy on a laptop.

## 4. What to delete regardless of path

The following remove themselves once any of Paths A, B, or C is
underway, but should be removed *now* to stop them from being cited
as evidence:

- **`core/dishworld.py` as headline evidence.** Demote to "Appendix:
  illustrative toy used to derive the policy form." Stop generating
  headline tables from it.
- **The `TinyMLP` "neural" experiments.** They fit hand-typed
  Bernoulli tables to themselves. Either remove or relabel as
  "sanity check: a 2-layer MLP can recover the parameters of the
  data-generating tables."
- **The scripted Meta-World adapter as a benchmark.** Either learn
  a policy or remove §10 entirely. The current section reads as a
  robotics result and is not one.
- **The Kona/Aleph contract section** until an external prover
  consumes the export. Until then, this is an unused serialisation
  format.
- **Forward-dated citations.** Mur-Labadia et al. (2026) with arXiv
  ID `2603.14482` is not a valid arXiv ID. Logical Intelligence
  (2026a, 2026b) should be either dated to the actual page-fetch
  year or removed. Assran et al. (2025) should be verified against
  arXiv.
- **The five-seed "robustness sweep" as evidence.** Five
  deterministic seeds is not statistics. Replace with bootstrap CIs
  over episodes, or remove the section.
- **The action-grounding challenge thresholds in their current
  form.** Either freeze the thresholds before the next run (commit
  them, version them, log every adjustment), or drop the "challenge"
  framing and call it what it is: a current-run report.
- **The abstract.** Cut to one-third length. The current abstract
  has ~30 numeric assertions; a reader cannot tell which are
  tautologies over hand-typed tables and which are empirical
  findings.

## 5. Sequenced milestone plan

### Phase 0 — Honesty pass (2 weeks)

- Apply all deletions in §4.
- Rewrite the abstract.
- Mark every remaining numeric claim with a tag from `{tautology,
  toy-mechanism, real-empirical, protocol-only}`.
- Commit a frozen `expected_margins.json` for the action-grounding
  challenge.

**Exit criterion:** the paper, as it stands, makes no claim it
cannot back up with the existing code under the new tagging.

### Phase 1 — Sheaf-honest toy (4 weeks)

- Implement the Path B construction (§2.2) in the existing
  `dishworld`. Cover, vertex stalks, edge stalks, learned
  restrictions, sheaf Laplacian, reported $\dim H^0, \dim H^1$.
- Run the decisive ablation (§2.3) in the toy.
- Construct one structural-$H^1$ task (Möbius-style overlap) and
  report whether the implementation detects it.

**Exit criterion:** either the sheaf model beats the scalar
ablation in the toy and the cohomology dimensions move as
predicted, or it does not and the paper pivots to Path A only.

### Phase 2 — Path A in MuJoCo (3 months)

- Build the environment (§1.2).
- Implement the encoder, posterior, probe-selection policies, and
  baselines.
- Run with paired-bootstrap statistics (§1.3).

**Exit criterion:** the Path A acceptance criteria are met or
falsified on at least one task.

### Phase 3 — Path B on the real environment (6 months)

- Only if Phase 1 produced a positive result.
- Combine the Phase 1 sheaf with the Phase 2 encoder.
- Re-run the ablation in the real environment.

**Exit criterion:** Path B acceptance criteria are met or falsified
in MuJoCo.

### Phase 4 — Robot policy benchmark (3–6 months)

- LIBERO or DROID via the existing manifest path
  (`robot/manifest_protocol.py` already validates the inputs).
- Use the encoder from Phase 2/3 as a frozen feature extractor
  feeding a policy head trained by BC or diffusion policy.
- Compare against the same policy head with passive features.

**Exit criterion:** the Path A acceptance criteria translated to
the robot setting are met or falsified.

### Phase 5 — Path C (optional, 12–18 months)

- Only after Phases 2 and 4 have established the mechanism is real.
- Requires GPU access. Defer the scope decision until the earlier
  phases have landed.

## 6. The honest minimum

If only one quarter is available and one person is doing the work:

1. Phase 0 (honesty pass).
2. Phase 1 (sheaf-honest toy).
3. A reduced Phase 2: one MuJoCo task, one parameter to randomise
   (friction), pixel input, real CBF safety, three policies (no
   probe, entropy probe, VOI probe).

This converts the project from "position paper with toy
illustrations" into "method with one real ablation and one real
benchmark with confidence intervals." It is the smallest possible
delivery that makes the central claim non-trivial.

Everything else can wait, including the sheaf framing, the JEPA
suffix, and the cohomology vocabulary. They can come back once
something real is underneath them.

## 7. Standing rules

While this plan is in effect:

- No new claim enters the paper without a verifier and a
  bootstrap CI or an explicit `tautology` / `toy-mechanism` /
  `protocol-only` tag.
- No table generated from `ACTION_MODEL` / `PROBE_LIKELIHOOD` is
  cited as evidence for anything except "the policy expression is
  correctly implemented."
- No reference is added without checking that the cited work
  exists and is published in the year stated.
- Negative results are reported in the body, not relegated to
  appendix smoke tests.
- The evidence matrix is regenerated and committed (or its current
  hash recorded) every time the paper changes.

## 8. Update: hypothesis-driven Phase 0 and Phase 6 (JEPA augmentations)

### Phase 0 completed

The hypothesis-driven Phase 0 has been executed. See
`docs/HYPOTHESIS_RESULTS.md` for the full results of H1-H5. Summary:

- H1 confirmed (obstruction gate is operationally vacuous)
- H2 falsified (active vs entropy survives 50-seed bootstrap)
- H3 confirmed (trained MLP matched by frozen random projection)
- H4 failed in unexpected direction (sheaf gluing actively hurts on
  categorical regimes despite reducing coboundary energy 10x)
- H5 inconclusive on the toy (variance-limited; bisim hurts at chosen
  weight, viability shows positive trend)

The original Phase 0 ("honesty pass" of deletions) was performed as a
single replacement: `paper/PAPER.md` was overwritten with the
augmentation-typology revision after H1-H5 landed. The original P-JEPA
paper content is preserved in git history at commit `bd7da45` and
earlier (`git show bd7da45:paper/PAPER.md`). A follow-up session can
still perform surgical edits to recover sections from the original if
needed.

### Phase 6 — JEPA-augmentation typology (new)

This phase replaces Phases 2-5 of the original plan with a tighter
target: implement the mathematical commitments from the original
paper as auxiliary losses on top of stock JEPA, rather than as a
replacement architecture.

The toy infrastructure landed in this session:

- `simulation/pjepa_sim/jepa_toy/` — NumPy JEPA with toggleable
  intervention, bisim, active masking, viability augmentations.
- `simulation/pjepa_sim/representation/sheaf_toy.py` — real cellular
  sheaf construction (learned cover, restriction maps, coboundary,
  Laplacian, H^0 / H^1).
- `docs/JEPA_AUGMENTATIONS.md` — PyTorch design specs for each
  augmentation, with proposed V-JEPA-scale evaluation protocols.
- `paper/PAPER.md` — paper rewritten as augmentation typology
  (the original 80KB P-JEPA paper is in git history at `bd7da45`).

The next concrete steps for Phase 6:

1. **Port the toy losses to PyTorch on top of a public V-JEPA
   reference implementation.** Start with intervention + composition
   consistency. Use Bardes et al. (2024) reference code or a public
   reimplementation.
2. **Pretrain ablations at small scale.** SSv2 sub-sample
   (~10K-50K clips), matched FLOPs across variants, ViT-S or ViT-B
   encoder. ~2-3 weeks per ablation on 1-2 A6000-class GPUs.
3. **Evaluate frozen-feature linear probe** on SSv2 action recognition
   AND held-out intervention-prediction task. Bootstrap CIs over
   evaluation episodes.
4. **Add sheaf consistency on overlapping clips fourth.** Conditional
   on the H4-positive prediction (continuous overlapping data should
   help; categorical does not).
5. **Add bisimulation with curriculum tuning fifth.** Needs working
   intervention head first.
6. **Add viability head when downstream is safety-critical** (e.g.,
   robot policy on LIBERO).

The success criterion is: at least one augmentation, on at least one
ablation, beats stock V-JEPA at matched FLOPs on at least one
intervention-relevant downstream metric with non-overlapping
bootstrap CIs. Negative results are reported honestly per the
standing rules above.

Expected cost: 3-6 months for one researcher with 1-2 GPUs.

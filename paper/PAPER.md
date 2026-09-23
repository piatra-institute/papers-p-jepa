---
title: |
  P-JEPA:\
  JEPA Augmentations from Embodied and Causal Mathematics
author: PIATRA . INSTITUTE
date: May 2026
---

## Abstract

P-JEPA was proposed as a replacement for the homogeneous target embedding of Joint Embedding Predictive Architectures (JEPA): a sheaf-valued predictive state over a stratified interaction space. Its reference implementation computes a posterior-weighted variance in place of a coboundary and never places the sheaf in a training loop. We recast each piece of the proposal's embodied and causal mathematics (intervention prediction, bisimulation, active masking, viability, sheaf consistency, composition consistency) as an auxiliary loss, head, or sampler that plugs into a stock JEPA training loop and can be ablated. Five preregistered hypothesis tests on the dishworld toy and the reference code give the following results. The obstruction gate never fires on any of the five reported suites, so the full P-JEPA agent is numerically identical to a plain value-of-information agent. Value-aware active probing beats entropy probing across 50 seeds (paired bootstrap CI95 [+0.009, +0.017]). A frozen random projection of equal width matches the trained intervention encoder (CI95 [+0.000, +0.0045]). A genuine cellular sheaf reduces coboundary energy tenfold but chooses actions slightly worse than the raw cover (CI95 [−0.005, −0.004]). On a NumPy JEPA with toggleable auxiliary losses, the viability head shows a positive trend (CI95 [−0.007, +0.10]) and bisimulation at weight 0.3 hurts (CI95 [−0.23, −0.03]). The toy is at its variance limit, so these are directional signals. A typology mapping each augmentation to the data structure it suits, and a priority order for V-JEPA-scale ablation, are offered as conjecture; the toy partly contradicts the order, finding the first-ranked intervention loss neutral and the last-ranked viability head the only positive trend.


## 1. Introduction

Joint Embedding Predictive Architectures learn representations by predicting the embedding of a masked target from the embedding of a context (LeCun, 2022; Assran et al., 2023; Bardes et al., 2024). P-JEPA was proposed as a new architecture in this family: a sheaf of predictive affordance models on a stratified interaction space, trained jointly with intervention, viability, and composition objectives. The mathematical objects involved (sheaves, coboundaries, cohomology, viability kernels) are sound. The reference implementation, however, contains a scalar posterior-weighted variance and a finite-state value-of-information solver, with no cellular sheaf, no JEPA encoder, and no comparison against any JEPA variant. A detailed critique is kept in `docs/CRITIQUE.md`, and the tests that settle its main points are reported in `docs/HYPOTHESIS_RESULTS.md`.

We adopt a plug-in formulation. JEPA is the substrate, and each piece of mathematics from the proposal becomes an auxiliary loss or head added to a stock JEPA training loop and evaluated by ablation. The contribution is correspondingly narrower: a typology of when each augmentation matches the structure of the data, together with a set of preregistered tests on a toy and on the reference code. The narrower claim can be tested experimentally, which the architectural claim could not.

The Meta-World adapter and the formal contract interface of the original proposal remain as supporting material in `docs/ARCHITECTURE.md` and the simulation code.


## 2. Intervention-Sufficient Representations

The representational criterion of the proposal is retained. A representation is *intervention-sufficient* at tolerance $\varepsilon$ if it preserves the information needed to predict the outcomes of arbitrary admissible actions:

$$
D\bigl(\mathbb{P}_A(o_{t+1:t+k}, v_{t+1:t+k} \mid h_t, do(\alpha)),\,
       \mathbb{P}_A(o_{t+1:t+k}, v_{t+1:t+k} \mid s_t, do(\alpha))\bigr) \leq \varepsilon.
$$

The criterion is the predictive-state criterion of Littman & Sutton (2001) extended with viability $v$ and interventional semantics (Pearl, 2009). For a JEPA augmentation its operational content is that $s_t$ must support predictions $\hat y_\alpha$ for each $\alpha$ in a chosen test bank, which is the intervention loss of Section 3. The remaining components of the proposal's mathematical stack (bisimulation, sheaf consistency, viability, composition, active perception) act as *secondary constraints* on the representation, each adding a different inductive bias to the predictive-state core.

The proposal combined these into a single objective with five auxiliary terms. Here they are five separable augmentations, each with its own ablation.


## 3. Augmentations as Loss Terms

Each augmentation is a loss or sampler that plugs into a stock JEPA training step. The full PyTorch signatures, their insertion points, and the proposed large-scale evaluations are in `docs/JEPA_AUGMENTATIONS.md`; the table below gives the names and loss equations.

| Name | Loss / sampler | Inductive bias | Where it matters most |
|--------|--------------------|----------|----------|
| Intervention | $\mathcal{L}_{do} = \mathbb{E}\|h_\psi(f_\theta(x), e(\alpha)) - y_\alpha\|^2$ | encoder must predict action outcomes | action-conditioned downstream tasks |
| Bisimulation | $\mathcal{L}_{\text{bisim}} = \mathbb{E}\bigl\|\|f(x)-f(x')\| - \mathbb{E}_\alpha D(\hat y_{x,\alpha}, \hat y_{x',\alpha})\bigr\|$ | latent metric anchored to outcome metric | tasks where visual similarity disagrees with action similarity |
| Active masking | $m^* = \arg\max_m \mathrm{Var}_k[g_\phi^{(k)}(f_\theta(x \odot m), m)]$ | hard-example mining via ensemble disagreement | sample-efficient pretraining |
| Viability head | $\mathcal{L}_{\text{viab}} = \mathbb{E}(\sigma(b_\psi(f(x), e(\alpha))) - u)^2$ | latent linearly separates unsafe states | safety-critical downstream policies |
| Sheaf consistency (on overlap) | $\mathcal{L}_{\text{glue}} = \mathbb{E}_{\Omega_{ij}}\|\rho_{i,ij}(f(x_i)) - \rho_{j,ij}(f(x_j))\|^2$ | adjacent clips encode coherently | video pretraining with overlapping windows |
| Composition consistency | $\mathcal{L}_{\text{comp}} = \mathbb{E}\|g(g(s, \alpha_1), \alpha_2) - g(s, \alpha_1 \circ \alpha_2)\|^2$ | predictor is associative under action composition | multi-step latent planning |

The bisimulation term follows the bisimulation-metric literature (Ferns et al., 2011; Zhang et al., 2021). The viability head is a learned analogue of a control barrier function (Ames et al., 2019). The sheaf-consistency term penalizes the coboundary of a cellular sheaf on the overlap graph (Hansen & Ghrist, 2019; Robinson, 2017). The intervention loss targets the causal confusion that arises when a representation encodes correlates of outcomes instead of their causes (de Haan et al., 2019).

The full loss is a weighted sum:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{JEPA}}
+ \lambda_{do}\,\mathcal{L}_{do}
+ \lambda_{\text{bisim}}\,\mathcal{L}_{\text{bisim}}
+ \lambda_{\text{viab}}\,\mathcal{L}_{\text{viab}}
+ \lambda_{\text{glue}}\,\mathcal{L}_{\text{glue}}
+ \lambda_{\text{comp}}\,\mathcal{L}_{\text{comp}}.
$$

Active masking is absent from the sum because it acts as a sampler. The $\lambda$ values are augmentation-specific weights. The proposal wrote this sum as a single objective $\mathcal{L}_{P\text{-JEPA}}$; treating the terms as separable allows ablation, by setting individual $\lambda$ values to zero and measuring downstream performance.


## 4. NumPy JEPA Toy on Dishworld

`simulation/pjepa_sim/jepa_toy/` implements a small NumPy JEPA on dishworld contexts: 6-dimensional inputs (four sensor features and a two-dimensional one-hot visual feature) generated from four hidden regimes. The model has a two-layer MLP encoder, an EMA target encoder, a mask predictor, an optional outcome predictor for the intervention loss, and an optional viability head, trained with Adam and manual backpropagation. Each seed trains in seconds.

The toy serves two purposes. The first is gradient verification: each auxiliary loss can be enabled and its loss curve checked for monotonic decrease, a correctness check before any V-JEPA-scale port. The second is directional evidence. The downstream evaluation is the regime-cluster and action-utility evaluation used throughout the simulation. A toy advantage suggests that an effect may reproduce at scale, and a toy disadvantage suggests that the augmentation's inductive bias may not suit this data type. Both readings are weak because the toy is at its variance limit, but they cost far less than GPU time.

A V-JEPA result requires PyTorch, a video dataset, matched compute, and frozen-feature linear probing against I-JEPA and V-JEPA baselines, none of which exists in this repository. The toy connects the mathematical form of each augmentation to a specification that can be tested at scale.


## 5. Preregistered Hypothesis Tests

Each hypothesis was registered with a binary pass/fail criterion before the experiment ran. Results are recorded in `docs/HYPOTHESIS_RESULTS.md`, with JSON artifacts in `simulation/output/experiments/`. The five verdicts are summarized below.

| # | Preregistered hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | the obstruction gate is a no-op | confirmed | the gate never fires; the two agents are bit-identical |
| H2 | active vs entropy probing is seed noise | rejected | active beats entropy by $+0.0130$, CI95 $[+0.0094, +0.0166]$, 41/50 seeds |
| H3 | the trained encoder matches a frozen random projection | confirmed | score delta CI95 $[+0.000, +0.0045]$ |
| H4 | the sheaf framing is decorative | criterion failed in the confirming direction | the glued centres score below scalar, CI95 $[-0.005, -0.004]$ |
| H5 | the JEPA augmentations help on the toy | not supported | no augmentation has a CI95 strictly above zero |

### H1: Obstruction Gate

The reference agents `p_jepa_stack` and `active_psr_probe` call the same `_decision_probe_result` function with different `use_obstruction_gate` flags. The gate short-circuits a probe when obstruction falls below `spec.sheaf_threshold`. **Result: PASS.** Across all five suites the initial obstruction (0.15–0.26) lies well above the threshold (0.06), so the gate never fires, and the two agents agree to floating-point precision. The distinction between "the P-JEPA stack" and "the active PSR probe" has no operational content on the reported suites.

### H2: Active Versus Entropy Probing

A 5-seed sweep in the proposal reported that value-aware active probing beats entropy probing by 0.005, with one seed favouring entropy. **Result: FAIL** (hypothesis rejected). With 50 seeds and 10000 paired bootstrap resamples, active probing beats entropy probing by a mean of +0.0130, CI95 [+0.0094, +0.0166], with 41 of 50 seeds favouring active probing. The 5-seed sweep had the right sign and too few seeds to resolve it. The claim that value-aware probing beats entropy probing is supported and should be reported with this interval.

### H3: Trained Encoder Versus Frozen Random Projection

The neural P-representation was claimed to recover regimes through a learned representation. **Result: PASS** (hypothesis confirmed). Across 10 seeds the trained MLP reaches a risk-adjusted score of 0.802 and cluster purity 1.000; a frozen random TinyMLP of identical width reaches 0.800 and 0.994. The paired CI95 on the difference is [+0.000, +0.0045] for score and [+0.000, +0.018] for purity. Clustering of linearly separable Bernoulli sources does the work, and gradient training adds nothing measurable; the mechanism is test-vector clustering, and the "neural" label is decorative.

### H4: Cellular Sheaf Versus Scalar Cover

A cellular sheaf is implemented in `simulation/pjepa_sim/representation/sheaf_toy.py`: a learned cover ($K=6$ k-means clusters), a 1-skeleton with 8.95 edges on average, learned linear restriction maps with ridge regularization, the assembled coboundary $\delta_0$, and the sheaf Laplacian $L_0 = \delta_0^\top \delta_0$, with mean $\dim H^0 = 9.85$ and $\dim H^1 = 42.3$ on the 1-skeleton. **Result: FAIL on the preregistered criterion, in the direction that strengthens the hypothesis.** Gluing lowers coboundary energy tenfold (0.354 to 0.0345), so the construction does what the mathematics says it should. The glued centres nevertheless score 0.798 against 0.802 for the scalar cover, with CI95 [−0.005, −0.004] entirely negative. On categorical hidden state the sheaf inductive bias slightly *hurts* downstream action choice. The criterion required a confidence interval containing zero; the interval excludes zero on the harmful side. The natural prediction is that on continuous overlapping data, such as V-JEPA temporal clips, where a global section exists to be recovered, the same mechanism should help (Section 6).

### H5: JEPA Augmentations on the Toy

Base JEPA and each augmentation (intervention, bisimulation, active masking, viability), plus all combined, were trained for 12 seeds × 6 variants on dishworld. **Result: FAIL** (no augmentation has a CI strictly above zero). The pattern behind the binary verdict is as follows.

| Variant | Mean score | Mean − base | CI95 |
|---|---:|---:|---|
| base JEPA | 0.591 | n/a | n/a |
| +intervention | 0.590 | −0.001 | [−0.127, +0.125] |
| +bisim | 0.462 | −0.130 | [−0.231, −0.027] |
| +active masking | 0.585 | −0.006 | [−0.059, +0.048] |
| +viability | 0.624 | +0.033 | [−0.007, +0.098] |
| +all | 0.477 | −0.114 | [−0.209, −0.023] |

Three findings stand out. Bisimulation at $\lambda = 0.3$ is mis-calibrated and hurts, with a CI that excludes zero on the negative side, consistent with reports that bisimulation objectives need careful curriculum tuning (Zhang et al., 2021). Viability shows a positive trend whose CI nearly excludes zero; its gain comes mostly from seeds on which base JEPA converges to a degenerate latent. Intervention and active masking are neutral at this scale. The seed-to-seed variance of the toy exceeds the augmentation effects.

These results do not falsify the augmentations. They show that the toy is at its detection limit and that bisimulation needs $\lambda$-curriculum work before it is tried at scale.


## 6. Conjectured Typology and Priority Order

Everything in Section 5 is a preregistered measurement with a binary verdict. The typology in this section is a conjecture: a set of hypotheses about which augmentation should help at V-JEPA scale, derived from the inductive-bias reasoning of Section 3 and only loosely constrained by the toy. It is untested at scale, and the toy evidence partly contradicts the priority order it proposes.

The H5 disagreement is the central caveat. The priority order ranks the intervention loss first. H5 is the only place where the toy can check that ranking, and there the intervention loss is neutral (mean −0.001, CI95 [−0.127, +0.125]). The only augmentation with a positive trend on the toy is the viability head (mean +0.033, CI95 [−0.007, +0.098]), which the order places last. Where the conjecture can be checked, the evidence points the other way. The ranking rests on two arguments: that the discrete Bernoulli dishworld is the wrong data type for the intervention bias, and that a variance-limited toy cannot resolve a real effect (Section 7). Both are arguments without measurements behind them, and the order should be read as a bet on inductive-bias reasoning that the available evidence does not yet support.

Combining the H1–H5 results with the inductive-bias analysis of Section 3 gives the following typology.

| Augmentation | Toy evidence | When to expect it to help at scale |
|---|---|---|
| Intervention loss | neutral on toy (variance-limited) | high gain on action-conditioned video (SSv2, robot rollouts); the inductive bias is well matched |
| Bisimulation | hurts at $\lambda=0.3$ | medium gain conditional on curriculum tuning and a well-trained intervention head |
| Active masking | neutral on toy | small-to-medium gain on representation pretraining |
| Viability head | positive trend, CI95 $[-0.007, +0.098]$ | high gain for safety-critical downstream tasks; low for plain recognition |
| Sheaf consistency | hurts on categorical regimes (H4) | *predicted to help* on continuous overlapping data (V-JEPA clips), conditional on the H4 boundary |
| Composition consistency | not tested on toy | medium gain on multi-step planning (V-JEPA 2); calibration of $k$-step rollouts |

The order below is the sequence we would advise a V-JEPA implementer to try, as a prioritized hypothesis. Each entry rests on inductive-bias reasoning, and the first is the one the toy contradicts.

- **Intervention loss first.** It has the highest *expected* gain on the inductive-bias argument, because the bias matches action-conditioned data. On dishworld it is neutral (H5), so the first place is a conjecture about scale and data type.
- **Composition consistency second.** It is cheap and targets multi-step rollouts.
- **Active masking third.** It is cheap and has precedent in hard-example mining.
- **Sheaf consistency on overlapping clips fourth.** This is conditional on the H4 boundary: the term is expected to work on continuous overlapping data and fails on categorical regimes.
- **Bisimulation fifth.** It requires a working intervention head and a careful $\lambda$ curriculum.
- **Viability last**, unless the downstream task is safety-critical, in which case first.

V-JEPA-scale ablation can falsify this order; the present evidence does not establish it.


## 7. Limitations

The toy is small. Dishworld has 4 categorical regimes, 6-dimensional contexts, and Bernoulli outcomes. The H5 toy is at its variance limit: base JEPA alone ranges from 0.44 to 0.80 across seeds. A toy negative is weak evidence against a gain at scale, and a toy positive is weak evidence for one. The PyTorch signatures of Section 3 and the evaluations proposed in `docs/JEPA_AUGMENTATIONS.md` exist because the decisive test is at V-JEPA scale, which this repository cannot run.

The tests do settle the status of several objects from the proposal. H1 shows that the obstruction gate is operationally vacuous. H4 shows that the sheaf framing is at best decorative and at worst slightly harmful on categorical regimes. H3 shows that the "neural" framing is decorative. H2 confirms the active-probing advantage over entropy probing, and H5 gives mixed directional evidence on the auxiliary losses.

The hidden-regime table, the Meta-World adapter, and the formal contract interface of the proposal are left untouched by these tests. They remain evidence that the value-of-information component of the P-JEPA stack works, since H1 shows that the full stack reduces to exact Bayesian value of information, and they bear on neither the sheaf nor the "neural" framing.

No augmentation is shown to beat stock V-JEPA at scale. The results establish that the augmentations have well-defined mathematical forms and correct NumPy implementations, that one (viability) shows a positive trend even on a variance-limited toy, and that one (bisimulation) needs tuning before scale. The priority order of Section 6 is not established, and on the one ranking the toy can check (H5, intervention loss first) the toy disagrees.

The scope excludes a new architecture and a foundation model. There is no benchmark against I-JEPA, V-JEPA, V-JEPA 2, or any video foundation model, and no learned robot controller, real perception, language grounding, or end-to-end neural sheaf. The cellular sheaf in `representation/sheaf_toy.py` is the only sheaf in the project, and on the one dataset on which it was tested (dishworld) it produced a negative result. That construction has a learned cover, learned linear restrictions, an assembled coboundary, a sheaf Laplacian, and reported cohomology dimensions, and it is ablated on a representation-learning task; the H4 result is therefore an empirical finding about the limits of sheaf-style coherence as a representation-learning prior on categorical data, beyond an absence of evidence.


## 8. Conclusion

Each embodied or causal commitment of the P-JEPA proposal has been given a loss-function form, a PyTorch signature, a toy-scale gradient-flow check, and a preregistered V-JEPA-scale evaluation protocol. The validated results are the Section 5 battery: the obstruction gate is inert, value-aware probing beats entropy probing, the trained encoder is matched by a frozen random projection, a genuine cellular sheaf slightly hurts action choice on categorical data, and no auxiliary loss clears zero on the toy, with viability trending positive and bisimulation at the chosen weight harmful. The typology and priority order of Section 6 are a research agenda, which H5 partly contradicts. Running even one augmentation at V-JEPA scale on Something-Something V2 or DROID is the natural next test. The JEPA toy, the hypothesis-test framework, the sheaf construction, and the design specifications needed for it are in the repository.


## 9. Reproducibility

All experiments use `uv` and run from `simulation/`:

```bash
uv run python -m pjepa_sim.experiments.h1_obstruction_gate
uv run python -m pjepa_sim.experiments.h2_seed_sweep_bootstrap
uv run python -m pjepa_sim.experiments.h3_frozen_random_baseline
uv run python -m pjepa_sim.experiments.h4_sheaf_vs_scalar
uv run python -m pjepa_sim.experiments.h5_jepa_augmentations
```

Each writes a JSON artifact to `output/experiments/` and prints a single PASS/FAIL line. The local audit (`uv run python -m pjepa_sim.cli.verify_all`) continues to pass; the experiments are additions to the codebase.

`docs/HYPOTHESIS_RESULTS.md` records which hypotheses passed, which failed, and the decisions the results imply. `docs/JEPA_AUGMENTATIONS.md` is the PyTorch design document: it gives the loss signatures, the insertion point of each loss in a V-JEPA reference implementation, and the success criterion that would constitute a positive result for each augmentation.

## References

- Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., & Tabuada, P. (2019). Control barrier functions: theory and applications. *European Control Conference*.
- Assran, M. et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA). arXiv:2301.08243.
- Bardes, A., et al. (2024). Revisiting feature prediction for learning visual representations from video (V-JEPA). arXiv:2404.08471.
- de Haan, P., Jayaraman, D., & Levine, S. (2019). Causal confusion in imitation learning. *NeurIPS*.
- Ferns, N., Panangaden, P., & Precup, D. (2011). Bisimulation metrics for continuous Markov decision processes. *SIAM Journal on Computing*, 40(6).
- Friston, K. (2010). The free-energy principle. *Nature Reviews Neuroscience*, 11.
- Hansen, J., & Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. *Journal of Applied and Computational Topology*, 3.
- LeCun, Y. (2022). A path towards autonomous machine intelligence. v0.9.2.
- Littman, M. L., & Sutton, R. S. (2001). Predictive representations of state. *NeurIPS*.
- Pearl, J. (2009). *Causality*. 2nd edition. Cambridge.
- Robinson, M. (2017). Sheaves are the canonical data structure for sensor integration. *Information Fusion*, 36.
- Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). A reduction of imitation learning to no-regret online learning. *AISTATS*.
- Zhang, A., McAllister, R., Calandra, R., Gal, Y., & Levine, S. (2021). Learning invariant representations for reinforcement learning without reconstruction (deep bisimulation). *ICLR*.

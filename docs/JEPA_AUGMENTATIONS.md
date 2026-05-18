# JEPA Augmentations from Embodied / Causal Mathematics

This document is the bridge between the paper's mathematical commitments
and a real V-JEPA implementation. The current paper introduces several
mathematical objects (intervention calculus, bisimulation, viability,
active perception, sheaf consistency, skill composition) and then
illustrates them in a 4-state toy. None is wired into a JEPA training
loop. This document specifies how each one *would* be wired.

For each augmentation it gives:

1. The math (the loss term in equation form).
2. The PyTorch signature of the loss as it would be added to a public
   V-JEPA reference implementation.
3. Where it plugs in (which forward pass, which optimizer, what
   conditioning).
4. The toy result from `experiments/h5_jepa_augmentations.py`.
5. The proposed real evaluation: dataset, baselines, success criterion.

Two facts to keep in mind throughout:

- The H4 experiment showed that on **categorical** hidden state
  (dishworld), sheaf gluing *hurts* downstream task value while still
  reducing coboundary energy 10x. The math is correct; the inductive
  bias is wrong for that data type. Each augmentation below has an
  analogous boundary condition where the inductive bias may or may not
  match the data.
- The H5 toy is at its variance limit: base JEPA itself ranges
  0.44-0.80 across seeds on dishworld, so an augmentation effect under
  ~0.15 cannot be cleanly detected with <50 seeds. The toy is a sanity
  check for *implementation correctness*, not a quantitative ranking.
  The real ranking will come from V-JEPA-scale runs.

Code references throughout point to
`simulation/pjepa_sim/jepa_toy/{model,losses,training,eval}.py`.

---

## A. Intervention loss ($\mathcal{L}_{do}$)

### Math

Given context $x$, a discrete intervention $\alpha$ from a finite
vocabulary, and the observed post-intervention outcome $y_\alpha$
(typically a low-dimensional vector — a robot action label, a frame
transformation, an applied physical perturbation):

$$
\mathcal{L}_{do} = \mathbb{E}_{(x, \alpha, y_\alpha) \sim \mathcal{D}_{\text{int}}}
\bigl\lVert h_\psi\bigl(f_\theta(x),\, e(\alpha)\bigr) - y_\alpha \bigr\rVert_2^2
$$

where $f_\theta$ is the JEPA encoder, $e(\alpha)$ is an action
embedding (one-hot or learned), and $h_\psi$ is a small intervention
head that predicts the outcome.

This breaks the imitation-learning causal confound: the encoder cannot
satisfy $\mathcal{L}_{do}$ by capturing temporal correlations alone.
It must encode features that are causally relevant to action outcomes.

### PyTorch signature

```python
class InterventionHead(nn.Module):
    def __init__(self, latent_dim: int, n_actions: int, hidden_dim: int, outcome_dim: int):
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, outcome_dim),
        )

    def forward(self, latent, action_idx):
        return self.head(torch.cat([latent, self.action_embed(action_idx)], dim=-1))


def intervention_loss(encoder, intervention_head, x, action_idx, outcome):
    return F.mse_loss(intervention_head(encoder(x), action_idx), outcome)
```

### Where it plugs in

Add to the JEPA training loop alongside the mask-prediction loss:

```python
loss_jepa = mask_prediction_loss(...)
loss_do = intervention_loss(encoder, int_head, x, alpha, y_alpha)
loss = loss_jepa + lambda_do * loss_do
loss.backward()
```

The intervention head is a new optimizer group. Encoder gradients sum
across both losses. Lambda starts at 1.0; tune on a small validation
set.

### Toy result (H5)

Implemented in `jepa_toy/losses.py::intervention_loss`. On dishworld
the per-epoch loss decreases monotonically (0.18 → 0.04 over 500
epochs, confirming the gradient routes correctly). Across 12 seeds the
score delta vs base JEPA is small with CI containing zero — but the
toy is variance-limited.

### Proposed real evaluation

Dataset: **Something-Something V2** with its 174 action templates.
Each video is naturally an $(x, \alpha, y_\alpha)$ triple: the first
$k$ frames are the context, the action template is $\alpha$, the
remaining frames condition the outcome embedding.

Baselines: stock V-JEPA at matched FLOPs, V-JEPA + auxiliary action
classification head.

Success criterion: linear probe on SSv2 action recognition stays
within ±1% of stock V-JEPA AND held-out *intervention prediction*
("given the first $k$ frames and a candidate action label, predict the
final-frame embedding") improves by ≥5% with non-overlapping bootstrap
CIs over 500 evaluation episodes.

Expected gain: medium-high. The setting matches the inductive bias.

---

## B. Bisimulation regularizer ($\mathcal{L}_{\text{bisim}}$)

### Math

For two contexts $x, x'$ and an action distribution $\rho(\alpha)$:

$$
\mathcal{L}_{\text{bisim}} = \mathbb{E}_{x, x'}
\Bigl\lvert
\lVert f_\theta(x) - f_\theta(x') \rVert -
\mathbb{E}_{\alpha \sim \rho}\bigl[
D\bigl(\hat y_{x,\alpha},\, \hat y_{x',\alpha}\bigr)
\bigr]
\Bigr\rvert
$$

where $\hat y_{x,\alpha}$ is the model's predicted outcome (from the
intervention head — bisim and intervention are natural co-implementers)
and $D$ is L2 on the outcome space.

This anchors the latent metric to action consequences. Two visually
similar contexts with different outcomes are pushed apart; two
visually different contexts with the same outcomes are pulled
together.

### PyTorch signature

```python
def bisimulation_loss(encoder, intervention_head, x_a, x_b, action_idx_samples):
    s_a = encoder(x_a)
    s_b = encoder(x_b)
    latent_dist = (s_a - s_b).norm(dim=-1)

    with torch.no_grad():
        outcomes_a = intervention_head(s_a.detach(), action_idx_samples)
        outcomes_b = intervention_head(s_b.detach(), action_idx_samples)
        target_dist = (outcomes_a - outcomes_b).norm(dim=-1).mean(dim=0)

    return (latent_dist - target_dist).abs().mean()
```

### Where it plugs in

Sample pairs from a buffer of recent training examples. Detach the
target through the intervention head so the bisim loss only updates
the encoder.

```python
loss = loss_jepa + lambda_do * loss_do + lambda_bisim * loss_bisim
```

### Toy result (H5)

Implemented in `jepa_toy/losses.py::bisimulation_loss`. The loss
decreases. The toy score is *worse* than base on average (mean delta
−0.11, CI [-0.23, +0.01]). Likely explanation: the bisim weight (0.3)
is too aggressive relative to the JEPA loss on this small toy,
collapsing the encoder toward a degenerate metric. At V-JEPA scale
with larger latent dimensions and a curriculum on $\lambda_{\text{bisim}}$
this is less likely.

### Proposed real evaluation

Dataset: a robot-video dataset with action labels (DROID, RoboMimic,
or a Meta-World rollout buffer). The bisim regularizer needs the
intervention head to be well-trained first, so this is naturally a
*two-stage* training: stage one is JEPA + intervention; stage two adds
bisim with a low initial weight.

Baselines: JEPA + intervention without bisim. Stock V-JEPA with the
same compute spent on more data.

Success criterion: downstream policy success rate on a held-out task
distribution improves by ≥3% with non-overlapping CIs. Or: the latent
distance between two contexts predicts the actual outcome distance
with $R^2 > 0.5$ on held-out pairs.

Expected gain: medium. Conditional on intervention head being well-trained.

---

## C. Active masking

### Math

Instead of randomly masking, choose masks that maximise predictor
uncertainty:

$$
m^* = \arg\max_{m \in \mathcal{M}}
\mathrm{Var}_{k=1..K}\bigl[ g_\phi^{(k)}\bigl(f_\theta(x \odot m),\, m\bigr) \bigr]
$$

where $g_\phi^{(k)}$ are $K$ predictor heads (or a single head with
MC dropout). Equivalent to expected obstruction reduction ($\S 6$ of
the paper) with "patch mask" in place of "physical probe."

### PyTorch signature

```python
class MaskPredictorEnsemble(nn.Module):
    def __init__(self, latent_dim, mask_dim, hidden_dim, k_predictors=3):
        super().__init__()
        self.predictors = nn.ModuleList([
            MaskPredictor(latent_dim, mask_dim, hidden_dim)
            for _ in range(k_predictors)
        ])

    def disagreement(self, latent, mask):
        outputs = torch.stack([p(latent, mask) for p in self.predictors])
        return outputs.var(dim=0).sum(dim=-1)


def choose_active_mask(encoder, ensemble, x, n_candidates):
    candidates = [random_mask() for _ in range(n_candidates)]
    latent = encoder(x * candidates[0])  # encoder shared
    scores = [
        ensemble.disagreement(encoder(x * m), m)
        for m in candidates
    ]
    return candidates[scores.index(max(scores))]
```

### Where it plugs in

Replaces the masking step in the JEPA data pipeline. Each ensemble
member is trained on the same loss against the chosen mask. They
diverge through random initialisation and any dropout in their heads.

### Toy result (H5)

Implemented in `jepa_toy/training.py::choose_active_mask` as
hard-example mining (n_candidates=4) using the current model's own
prediction error as the uncertainty proxy. On dishworld the delta vs
random masking is small (mean −0.03, CI [-0.12, +0.04]), again
inconclusive at this scale.

### Proposed real evaluation

Dataset: I-JEPA on ImageNet at 100% compute matched to baseline.

Baselines: I-JEPA with standard random block masking.

Success criterion: linear probe on ImageNet validation accuracy
improves by ≥0.3% with non-overlapping CIs, OR linear probe accuracy
matched at 75% of training compute.

Expected gain: small-to-medium. Adjacent literature on active
self-supervised learning (active MoCo, hard negative mining) suggests
~0.5-1% gains are plausible.

---

## D. Viability head

### Math

A small head $b_\psi : \mathbb{R}^{d_{\text{latent}}} \times \mathcal{A} \to [0, 1]$
predicts the unsafe-failure probability for a (state, action) pair:

$$
\mathcal{L}_{\text{viab}} = \mathbb{E}_{(x, \alpha, u) \sim \mathcal{D}_{\text{safety}}}
\bigl[\bigl(\sigma\bigl(b_\psi(f_\theta(x), e(\alpha))\bigr) - u\bigr)^2\bigr]
$$

where $u \in \{0, 1\}$ is the observed unsafety indicator. The encoder
gets gradient through $b_\psi$, so the latent learns to make unsafe
states linearly separable.

### PyTorch signature

```python
class ViabilityHead(nn.Module):
    def __init__(self, latent_dim, n_actions, hidden_dim):
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent, action_idx):
        return self.head(torch.cat([latent, self.action_embed(action_idx)], dim=-1)).squeeze(-1)


def viability_loss(encoder, viability_head, x, action_idx, unsafe_label):
    logits = viability_head(encoder(x), action_idx)
    return F.binary_cross_entropy_with_logits(logits, unsafe_label.float())
```

### Where it plugs in

Joint training with JEPA when the dataset has safety labels (robot
rollouts with success/failure flags, autonomous-driving incidents,
laboratory experiment near-misses).

### Toy result (H5)

Implemented in `jepa_toy/losses.py::viability_loss`. Trains cleanly.
On dishworld the score delta is small with CI [-0.06, +0.18]. Most
informative per-seed: seed 103 (base 0.44 → viability 0.80) and seed
105 (base 0.80 → viability 0.68). The head rescues failure cases but
can hurt successful ones. This is consistent with the viability head
acting as an *additional regulariser* that pulls the latent toward
unsafety-discriminative geometry — which is helpful when the JEPA
loss has converged to a degenerate basin and harmful when JEPA has
already found a good one.

### Proposed real evaluation

Dataset: a robot dataset with explicit failure annotations (DROID has
some; LIBERO + manual labelling is feasible).

Baselines: V-JEPA + intervention head trained as classifier on
success vs failure.

Success criterion: downstream policy (BC or diffusion) trained on
frozen JEPA features has lower unsafe-failure rate at matched success
rate, with non-overlapping CIs over 200 evaluation episodes.

Expected gain: medium-high for safety-critical downstream tasks. Low
for plain action recognition.

---

## E. Sheaf consistency on overlapping clips

### Math

For two clips $i, j$ from the same video with overlapping temporal
window $\Omega_{ij}$:

$$
\mathcal{L}_{\text{glue}} = \mathbb{E}_{(i, j) : \Omega_{ij} \neq \emptyset}
\bigl\lVert \rho_{i,ij}\bigl(f_\theta(x_i)\bigr) - \rho_{j,ij}\bigl(f_\theta(x_j)\bigr) \bigr\rVert_2^2
$$

where $\rho_{k,ij} : \mathbb{R}^{d_{\text{latent}}} \to \mathbb{R}^{d_{\text{edge}}}$
are *learned* restriction maps per overlap.

This is the operational version of $\|d\sigma\|^2$ from §5-6 of the
paper, now applied where the H4 result said it should be applied:
genuinely overlapping continuous data, not categorical hidden state.

### PyTorch signature

```python
class RestrictionMap(nn.Module):
    def __init__(self, latent_dim, edge_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, edge_dim, bias=False)
        # Initialise as a projection of the identity-truncated form.
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(edge_dim, latent_dim))

    def forward(self, latent):
        return self.linear(latent)


def sheaf_consistency_loss(encoder, rho_i, rho_j, clip_i, clip_j):
    s_i = encoder(clip_i)
    s_j = encoder(clip_j)
    return F.mse_loss(rho_i(s_i), rho_j(s_j))
```

### Where it plugs in

Sample clip pairs from the same video with overlap; share an encoder.
Restriction maps are per-overlap parameters but can be tied across all
edges (one global $\rho$) to start — that recovers the standard
"adjacent frames should encode consistently" intuition.

### Toy result (H4 on dishworld)

H4 result is *negative on categorical hidden state*: sheaf gluing
reduces coboundary energy 10× but decreases downstream score by 0.4%
(CI [-0.005, -0.004], excludes zero). See `representation/sheaf_toy.py`
and `experiments/h4_sheaf_vs_scalar.json`.

This is informative for the V-JEPA application: the H4 finding
predicts that sheaf gluing should help when the underlying data has
genuine continuous overlap structure (adjacent video frames) and
should *not* help when the data is piecewise-categorical (discrete
regime swaps). V-JEPA's temporal clips are the first kind.

### Proposed real evaluation

Dataset: V-JEPA pretraining on Kinetics-400, with overlapping temporal
clips (e.g., clip A = frames [0, 16], clip B = frames [8, 24]).

Baselines: stock V-JEPA with non-overlapping clips, V-JEPA with
overlapping clips but no consistency loss.

Success criterion: SSv2 linear probe accuracy improves by ≥0.5%
with non-overlapping CIs, OR temporal coherence metric on held-out
clip-pair similarity improves measurably.

Expected gain: small-to-medium. This is the H4-positive case but the
expected effect size is bounded by the fraction of training data with
genuine overlap structure.

---

## F. Composition consistency ($\mathcal{L}_{\text{comp}}$)

### Math

For action pairs $(\alpha_1, \alpha_2)$ and an action-conditioned
predictor $g_\phi$:

$$
\mathcal{L}_{\text{comp}} = \mathbb{E}_{x, \alpha_1, \alpha_2}
\bigl\lVert g_\phi\bigl(g_\phi(f_\theta(x), \alpha_1),\, \alpha_2\bigr)
- g_\phi\bigl(f_\theta(x),\, \alpha_1 \circ \alpha_2\bigr) \bigr\rVert_2^2
$$

The predictor should be associative under action composition. This
keeps multi-step latent rollouts calibrated.

### PyTorch signature

```python
def composition_loss(encoder, predictor, x, alpha_1_idx, alpha_2_idx, composed_idx):
    s = encoder(x)
    two_step = predictor(predictor(s, alpha_1_idx), alpha_2_idx)
    one_step = predictor(s, composed_idx)
    return F.mse_loss(two_step, one_step)
```

### Where it plugs in

Requires a composed-action vocabulary. For V-JEPA 2 with robot actions,
$\alpha_1 \circ \alpha_2$ can be the action that *would* take the
robot the same place in two steps as the composition.

### Toy result

Not implemented in H5 (dishworld has no natural action composition;
the existing `composition.py` benchmark is a separate setup with
engineered chain tables). Skipped for the toy ablation.

### Proposed real evaluation

Dataset: V-JEPA 2's robot planning rollouts on Meta-World or DROID.

Baselines: V-JEPA 2 without composition loss.

Success criterion: $k$-step planning success rate at $k=2, 4, 8$
improves with non-overlapping CIs. Calibration of $k$-step latent
predictions (measured by Brier score or MSE to actual $k$-step
embedding) improves.

Expected gain: medium for multi-step planning. Negligible for
single-step prediction.

---

## Priority ordering for V-JEPA implementation

Based on (a) inductive-bias match with V-JEPA's training setting,
(b) PyTorch implementation cost, (c) expected effect size:

1. **Intervention loss + action-conditioned head.** Highest expected
   gain. Most natural V-JEPA extension. Requires action-labelled data
   (SSv2 templates, robot actions).
2. **Composition consistency.** Cheap to add. Likely improves
   multi-step planning in V-JEPA 2 specifically.
3. **Active masking.** Cheap. Adjacent literature suggests modest
   gains on representation quality.
4. **Sheaf consistency on overlapping clips.** Conditional on H4-positive
   pattern. Worth testing once data pipeline supports overlapping
   clips.
5. **Bisimulation regularizer.** Requires a well-trained intervention
   head first. Two-stage training. Medium gain conditional on stage 1.
6. **Viability head.** Domain-specific (safety-critical applications).
   Lower priority for general representation learning.

Doing 1, 2, 3 together as a single PyTorch JEPA-variant
implementation is the minimum publishable contribution. Add 4 if
overlapping clips are easy to extract. Add 5, 6 when downstream tasks
warrant them.

## Standing limitations

- The toy ablation (H5) is variance-limited. A toy negative is not a
  scale negative. A toy positive is weak directional evidence.
- The "real evaluation" sections are proposed protocols, not run
  results. None of these augmentations has been tested against V-JEPA
  at scale. The success criteria are preregistered for the next
  session's GPU work.
- The mathematics around each augmentation is principled but the
  *combination* of multiple augmentations introduces interactions
  that are not theoretically characterised. Joint ablations may show
  effects that individual ablations do not.

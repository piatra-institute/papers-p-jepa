---
title: |
  P-JEPA:\
  Predictive Praxis and the\
  Local-to-Global Problem of Embodied Competence
author: PIATRA . INSTITUTE
date: May 2026
---

## Abstract

Joint-embedding predictive architectures learn by predicting target representations from context representations. In the canonical equation, a context encoder maps $x$ to $s_x$, a target encoder maps $y$ to $s_y$, and a predictor is trained so that $g(s_x,z)$ lies close to $s_y$ in representation space. The architecture avoids pixel-level reconstruction and therefore learns abstractions useful for image and video understanding. Physical competence requires an additional object. An agent acting in the world must represent action-conditioned consequences, viability constraints, body-specific reachability, and compatibility among local models valid only in particular contact, material, social, and task regimes. This paper formalises that object as a *P-representation*, where $P$ names praxis: the capacity to reach goals through viable, embodied, counterfactual action. A P-representation is a compressed predictive state sufficient for estimating the consequences of possible interventions under safety constraints. Its mathematical stack combines predictive-state tests, bisimulation-style quotients, causal interventions, viability, active probing, local-to-global consistency, and skill composition. One computable form is a sheaf of predictive affordance models over a stratified interaction space. Local sections are context-bound predictive control models; compatible global sections are coherent embodied competence; the coboundary $d\sigma$ measures local disagreement; cohomology records residual incompatibilities that cannot be removed by local corrections. P-JEPA is the learning architecture obtained by replacing a single homogeneous embedding with this sheaf-valued predictive state and by adding losses for restriction consistency, intervention, viability, reachability, and skill composition. The accompanying simulation implements a hidden-regime manipulation world in which visually identical objects differ in action consequences. Baselines that act from visible class or prior-average prediction succeed at $0.450$ and $0.753$ respectively. A pure obstruction-reduction policy succeeds at $0.851$ and reduces unsafe failure to $0.100$; the full P-JEPA stack succeeds at $0.853$, reduces unsafe failure to $0.081$, and raises risk-adjusted score from $0.452$ to $0.651$ relative to the prior predictive baseline. A benchmark sweep shows that viability-aware active probing improves all tested suites, including against posterior-entropy probing, in the costly-probe setting where pure obstruction reduction is penalised, and in a miscalibrated learned-section suite where the policy's belief model differs from the true world model. A separate representation-learning benchmark makes the action-grounding claim explicit: when visual cues shift between train and test, visual grouping scores $0.282$, prior averaging scores $0.453$, and unlabeled action-consequence grouping scores $0.802$, matching the oracle regime score. A local video-representation surrogate then compares a passive JEPA-like next-frame predictor with an action-conditioned representation: the passive model predicts future frames with mean absolute error $0.017$ but scores $0.282$ with action-regime purity $0.500$, while the action-conditioned representation scores $0.802$ with purity $1.000$; this is not a V-JEPA benchmark. A load-bearing real-video smoke test on the six official KTH sample AVI files gives the opposite warning: static appearance scores $0.896$, a passive next-frame descriptor scores $0.805$, and temporal motion scores $0.623$, so this sample split is appearance dominated and is not evidence for a P-JEPA video advantage. A neural intervention-encoder benchmark replaces engineered fingerprints with a small NumPy MLP trained from sampled intervention records over low-dimensional physical sensor observations and test identities; the learned predicted-test vector scores $0.802$ with purity $1.000$, beating appearance and prior baselines and matching the engineered reference. A sample-efficiency sweep varies sampled intervention repeats from $1$ to $10$ per context; the neural P-representation keeps purity $1.000$, stays within $0.000$ of the engineered reference at every tested budget, and reduces predicted-test mean absolute error from $0.051$ to $0.015$. A learned active-probing benchmark aliases dry with soapy and cracked with heavy in the initial sensor observation; value-aware probing raises score from $0.485$ to $0.698$, lowers unsafe failure from $0.143$ to $0.069$, and beats entropy probing at $0.678$. A boundary sweep shows why this is not an unconditional claim: with distinct initial sensors, no-probe and active policies both score $0.802$; with weak probes, active probing improves only from $0.485$ to $0.490$; with costly probes, the value-aware policy uses $0.625$ probes on average and still raises score to $0.647$. Across five deterministic seeds in the aliased-sensor setting, active probing beats no probing on every seed with minimum score margin $0.196$ and minimum unsafe-failure reduction $0.068$; its mean score is $0.689\pm0.013$, while its mean advantage over entropy probing is only $0.005$ and one seed favours entropy. A first rendered-pixel continuous-control benchmark replaces structured sensor vectors with $12\times 12$ images and discrete task actions with continuous 2D reach-controller rollouts; pixel active probing raises score from $0.515$ to $0.572$, reduces unsafe failure from $0.105$ to $0.089$, and remains far below the oracle score $1.000$. A formal contract-interface benchmark exports finite safety, branch-safety, obstruction, score, and probe-budget contracts for verification backends: under a local exhaustive checker, the P-JEPA stack satisfies $5/5$ suite contracts, entropy probing satisfies $1/5$, the prior baseline satisfies $0/5$, and the report explicitly records that no Kona or Aleph backend was executed. An online-cover benchmark constructs the same four action regimes incrementally from an unlabeled stream and again reaches score $0.802$ with purity $1.000$. A synthetic scaling sweep increases hidden action regimes from $4$ to $32$; action-consequence grouping keeps minimum purity $0.938$, reaches score $0.803$ in the $32$-regime case, and stays within $0.001$ of the oracle score, while appearance and prior baselines remain below $0.218$. A restriction-map gluing ablation then gives local action sections incompatible coordinate frames; identity/no-glue aggregation scores $0.572$ with overlap residual $0.384$, while learned restriction maps reduce residual to $0.017$ and score $0.796$, within $0.007$ of the hidden-regime oracle. A skill-composition benchmark requires a preparation skill to create an intermediate postcondition before a finishing skill can safely act; action-consequence grouping scores $0.829$ and selects the intended chains, while appearance grouping scores $0.438$ and prior averaging scores $0.421$. A preliminary Meta-World adapter benchmark gives the same qualitative signal in a continuous-control wrapper: obstruction-selected probing beats same-budget random and entropy probe baselines, and matches exhaustive random probing's risk-adjusted score while using one third as many probes. Learned-model variants estimate probe likelihoods and local action-effect sections from wrapper experience. With supervised regime labels, obstruction probing reaches score $0.813$ versus $0.800$ for exhaustive random probing and $0.560$ for same-budget random probing. With unlabeled context fingerprints clustered by action consequences, obstruction probing reaches $0.860$, again matching exhaustive random probing while using one probe rather than three. With raw unlabeled probe/action records, it reaches $0.860$ while beating entropy probing's $0.766$ and exhaustive random probing's $0.820$ score.


## 1. The missing variable in JEPA

LeCun's JEPA program begins from a clean self-supervised principle. Given a context $x$, a target $y$, a context encoder $f_\theta$, a target encoder $f_{\bar\theta}$, and a predictor $g_\phi$, the model minimises an energy

$$E_{\theta,\phi}(x,y,z) =
D\left(f_{\bar\theta}(y), g_\phi(f_\theta(x),z)\right),$$

where $z$ supplies mask, location, latent choice, action, or another conditioning variable. The target is representational. I-JEPA instantiates this principle for images by predicting target-block representations from a visible context block; V-JEPA and V-JEPA 2 extend the same direction into video, future-state prediction, and action-conditioned robot planning (LeCun, 2022; Assran et al., 2023; Bardes et al., 2024; Assran et al., 2025).

The equation has genuine force. It directs capacity away from local texture and toward the predictable structure of the scene. It also supplies a useful, if still imperfect, answer to the collapse problem in joint embedding systems: the target branch, predictor asymmetry, masking strategy, and update schedule help avoid trivial constant codes in practice. The result is a family of visual representations that can support downstream classification, object counting, depth estimation, video question answering after language-model alignment, and short-horizon physical prediction.

Physical intelligence asks a different representational question. An embodied system needs a state from which possible interventions can be evaluated. The relevant question moves from next-embedding prediction to the estimation of what will happen if the agent grasps here, tilts there, brakes now, yields to the cyclist, scrubs the underside, or changes contact geometry.

The missing variable is praxis. Praxis is action that remains answerable to the body, the environment, and the consequences of the action. A praxis representation must carry at least four kinds of information: what action sequences are available, what outcomes they are likely to produce, which outcomes preserve viability, and how local action models agree when contexts overlap. A visual embedding can contain some of this information implicitly. A physical agent needs it structurally.

The paper denotes this capacity by $P$. The letter is deliberately plain. $P$ names the structured capacity of an agent to act through a body in a world without leaving the conditions under which further action remains possible.

The contribution is threefold. First, the paper gives an intervention-sufficiency criterion for embodied representations. Second, it shows how predictive-state tests, causal interventions, viability, active probing, local-to-global consistency, and skill composition fit into one P-JEPA objective. Third, it supplies executable hidden-regime benchmarks in which obstruction-driven probing improves success, safety, or probe efficiency under the predicted conditions and fails when probe cost dominates. The sheaf is not the whole theory; it is the bookkeeping structure that makes local action models comparable.


## 2. P-representations

Let an agent's history at time $t$ be

$$h_t = (o_0,a_0,o_1,a_1,\ldots,o_t),$$

where $o_i$ are observations and $a_i$ are actions or motor commands. A representation is a compression

$$s_t = \phi(h_t).$$

For passive representation learning, the criterion for $s_t$ is usually predictive accuracy on future observations or future embeddings. For praxis, the criterion is intervention sufficiency.

\begin{definition}[P-representation]
For an agent $A$, a family of admissible action tests $\mathcal{T}$, an observation space $\mathcal{O}$, a viability variable $v_t \in [0,1]$, and a divergence $D$, a representation $s_t = \phi(h_t)$ is a P-representation at tolerance $\varepsilon$ when, for every relevant future action sequence $\alpha \in \mathcal{T}$,
$$
D\left(
\mathbb{P}_A(o_{t+1:t+k}, v_{t+1:t+k}\mid h_t, do(\alpha)),
\mathbb{P}_A(o_{t+1:t+k}, v_{t+1:t+k}\mid s_t, do(\alpha))
\right) \leq \varepsilon.
$$
\end{definition}

The definition says that the compression preserves the information needed to answer action-counterfactual questions under viability constraints. The intervention marker $do(\alpha)$ follows Pearl's distinction between observation and intervention (Pearl, 2009). Observing that plates often move after hands touch them is different from estimating what happens if this gripper applies this force at this point on this wet plate. A P-representation must support the second estimate.

This connects P-representations to Predictive State Representations, where the state of a controlled dynamical system is represented by predictions of future tests (Littman & Sutton, 2001; Singh et al., 2003; Boots, Siddiqi & Gordon, 2011). It also connects them to bisimulation metrics, which identify states by action-relevant consequences rather than by appearance (Ferns, Panangaden & Precup, 2011). The quotient principle is simple. Two physical situations should be represented together only when the relevant actions have the same outcome distributions under the task and viability constraints.

For a driving agent, two visually similar frames can differ in praxis state because one road surface contains black ice and the other does not. For a manipulation agent, two plates can share an object label and differ in praxis state because one is dry, one is wet, and one has a crack at the rim. For a mobile robot, two corridors can share visual geometry and differ in praxis state because one floor has low friction. The representation is action-indexed.

This gives the first correction to ordinary JEPA. Ordinary JEPA trains against a target embedding. P-JEPA trains against a family of action-conditioned future distributions, indexed by the tests the body can actually perform.


## 3. The mathematical stack of P

P-JEPA combines several mathematical commitments. None is sufficient alone.

| Component | Mathematical role | Failure if absent |
|---|---|---|
| Predictive state | represent state by answers to action tests | embeddings remain observational |
| Bisimulation quotient | merge histories by action consequences | visual similarity dominates praxis |
| Intervention calculus | distinguish seeing from doing | correlations become bad action models |
| Viability and reachability | preserve conditions for further action | goal reaching can destroy competence |
| Active perception | choose actions that improve state estimation | uncertainty remains passive |
| Sheaf consistency | glue local models on overlaps | local skills contradict each other |
| Skill composition | compose preconditions and postconditions | isolated skills do not form praxis |

The sheaf layer enters after the predictive-state and intervention layers. A local section is meaningful only because it predicts the consequences of tests; an obstruction is meaningful only because it changes which intervention should be taken next. P-JEPA is therefore closer to an active predictive-state architecture with local-to-global constraints than to a purely topological model.


## 4. The interaction space

Embodied action occurs in regimes. A robot hand approaching a plate, touching the rim, establishing a stable grasp, beginning to slip, colliding with the sink wall, and fracturing the plate occupies six different local worlds. The state variables, admissible actions, contact equations, failure modes, and recovery strategies differ across the six. Smooth control inside one regime does not remove the regime transition problem.

Let $\mathfrak{X}_A$ denote the agent-specific interaction space. A point records body configuration, relevant environment variables, contact mode, task context, partial history, and viability state. The space is naturally stratified:

$$\mathfrak{X}_A = \bigcup_i X_i,$$

where each stratum $X_i$ is a local interaction regime. The strata may correspond to no contact, point contact, stable grasp, slip, collision, breakage, fluid capture, social negotiation, or another domain-specific mode. Within a stratum, local models may be smooth enough for differential geometry and control. Across strata, transition maps carry discontinuities, guards, resets, hysteresis, and irreversible changes.

The viability set is a subset

$$K_A \subseteq \mathfrak{X}_A.$$

It contains the states from which the agent remains intact enough to continue acting. For dynamics

$$q_{t+1}=F(q_t,a_t,w_t),$$

with disturbances $w_t\in W$, viability theory supplies the sharper object:

$$\operatorname{Viab}(K_A) =
\{q_0 \in K_A : \exists \pi \ \forall (w_t)_{t\geq 0}\in W^{\mathbb{N}},
q_t \in K_A \text{ for all } t\}.
$$

A goal $G$ is praxis-reachable when a policy reaches $G$ while keeping the trajectory inside the viability kernel under the relevant disturbances. Hamilton-Jacobi reachability, control barrier functions, hybrid systems, and geometric control theory give existing mathematics for parts of this condition (Aubin, 1991; Goebel, Sanfelice & Teel, 2012; Bansal et al., 2017; Ames et al., 2019; Bullo & Lewis, 2004).

The role of $\mathfrak{X}_A$ is to prevent a false universalism in representation. The same environment offers different affordances to different bodies. A wheeled robot, a suction gripper, a five-fingered hand, and a human driver inhabit different interaction spaces even when they occupy the same physical location. Praxis is substrate-relative.


## 5. The affordance sheaf

Physical competence is local before it is global. Dry grasp, wet tilted grasp, lane keeping on dry pavement, and black-ice recovery can require different local models. The question is whether these local models can be made compatible on the overlaps where regimes meet.

This is a local-to-global problem. A global competence exists only when local predictive affordance models agree on the interfaces where their contexts overlap. Sheaf theory gives the language for that agreement: local sections, restriction maps, compatible families, and global sections (Curry, 2014).

\begin{definition}[Affordance presheaf and sheaf]
For an agent $A$, an affordance presheaf is a functor
$$
\mathcal{F}_{\mathrm{aff}}:
\operatorname{Open}(\mathfrak{X}_A)^{op}
\rightarrow \mathbf{PredControl},
$$
where $\mathcal{F}_{\mathrm{aff}}(U)$ is the space of predictive control-affordance models valid on the local context $U \subseteq \mathfrak{X}_A$.
\end{definition}

It is a sheaf when compatible local sections on a cover have a unique amalgamation. In the learned case this axiom is approximate: the model minimises incompatibility rather than assuming perfect gluing.

The target category $\mathbf{PredControl}$ may contain predictive distributions, viable sets, reachable sets, local controllers, sensor models, and skill preconditions. A local section

$$\sigma_U \in \mathcal{F}_{\mathrm{aff}}(U)$$

is a model of what can be done in context $U$. A simple version is a predictive state

$$\sigma_U : \mathcal{T}_U \rightarrow \Delta(\mathcal{O}_U \times [0,1]),$$

mapping admissible action tests in $U$ to distributions over observations and viability scores. Richer versions include local dynamics, barrier certificates, tactile uncertainty, symbolic constraints, and controllers.

For $V \subseteq U$, the restriction map

$$\rho_{UV}: \mathcal{F}_{\mathrm{aff}}(U) \rightarrow \mathcal{F}_{\mathrm{aff}}(V)$$

sends the broader local model to its narrower interface. It may forget tests unavailable in $V$, marginalise outcome variables, project a controller onto lower-dimensional contact coordinates, or translate a visual-tactile model into the subset of variables shared by the two contexts. The restriction map is a modelling choice with empirical content. A model that drops friction at the wet-grasp interface is asserting that friction does not matter there. The action failures will audit the assertion.

Given a finite cover $\{U_i\}$ of a task domain, a family of local sections $\sigma = \{\sigma_i\}$ has coboundary

$$
(d\sigma)_{ij}
= \rho_{i,ij}(\sigma_i) - \rho_{j,ij}(\sigma_j)
\quad \text{on } U_i \cap U_j.
$$

The equality is literal when the stalks are vector spaces and approximate when the stalks carry probability distributions, controllers, or learned models. A coherent praxis representation satisfies

$$d\sigma = 0.$$

The zeroth cohomology gives the coherent sections:

$$H^0(\mathfrak{X}_A,\mathcal{F}_{\mathrm{aff}})
= \ker d_0.$$

The first cohomology is

$$
H^1(\mathfrak{X}_A,\mathcal{F}_{\mathrm{aff}})
=
\ker d_1/\operatorname{im} d_0.
$$

It records residual incompatibility modulo local corrections. The raw disagreement $d\sigma$ is the operational obstruction used for learning; the cohomology class is the invariant part that remains after admissible local adjustments.

For P, the reading is direct:

$$H^0 = \text{coherent embodied competence},$$

$$H^1 = \text{residual local-to-global incompatibility}.$$

Applied sheaf theory already uses this language for sensor integration. Robinson (2017) treats sheaves as a canonical data structure for heterogeneous sensor fusion, with cohomology measuring consistency among sources. Hansen and Ghrist (2019) lift graph Laplacians to cellular sheaf Hodge Laplacians, giving computable spectral tools for local-to-global consistency. Neural sheaf diffusion then imports learnable sheaves into graph learning (Bodnar et al., 2022), and recent surveys track the broader entry of sheaf methods into machine learning (Ayzenberg et al., 2025). P-JEPA applies the same mathematical commitment to embodied predictive control.


## 6. The P-JEPA objective

P-JEPA is obtained by replacing JEPA's homogeneous target representation with a sheaf-valued predictive praxis target. In a neural implementation, the local prediction term can retain the JEPA form:

$$
\mathcal{L}_{\mathrm{pred}}
=
\mathbb{E}_{U,\alpha}
D\left(
f_{\bar\theta}(y_{U,\alpha}),
g_\phi(f_\theta(h_U), e(U), e(\alpha), z)
\right),
$$

where $h_U$ is the local action-observation history inside context $U$, $\alpha$ is an action test, $e(U)$ encodes the local context, $e(\alpha)$ encodes the candidate intervention, and $y_{U,\alpha}$ is the observed future under that intervention or a target produced by an EMA branch from the corresponding future segment. A readout from the predicted representation parameterises the local section $\sigma_\theta(U,h_U,\alpha)$, including outcome and viability predictions.

The full objective adds four terms:

$$
\begin{aligned}
\mathcal{L}_{P\text{-JEPA}}
=\;&
\mathcal{L}_{\mathrm{pred}}
+ \lambda \|d\sigma_\theta\|^2
+ \mu \mathcal{L}_{do}
+ \nu \mathcal{L}_{\mathrm{viab}}
+ \eta \mathcal{L}_{\mathrm{comp}}.
\end{aligned}
$$

The sheaf term $\|d\sigma_\theta\|^2$ penalises disagreement of local predictive models on overlaps. Written explicitly,

$$
\|d\sigma_\theta\|^2
=
\sum_{i,j}
\mathbb{E}_{h\in U_i\cap U_j,\alpha\in\mathcal{T}_{ij}}
D_{ij}\left(
\rho_{i,ij}(\sigma_i)(h,\alpha),
\rho_{j,ij}(\sigma_j)(h,\alpha)
\right).
$$

It is the differentiable counterpart of asking whether local sections glue. In a vector-space sheaf it is the degree-zero Hodge energy associated with $\Delta_0 = d^\ast d$. In a probabilistic or controller-valued sheaf it becomes a divergence between restricted predictions, reachable sets, or controller outputs on shared interfaces.

The intervention term $\mathcal{L}_{do}$ forces the model to learn from actions rather than only from observation. It is estimated from self-experiments, teacher perturbations, policy rollouts, randomised micro-actions, and recovery episodes. Its role is to separate a regularity from a cause. A driving model that treats brake lights as the cause of braking has learned a useful correlation and a bad intervention model. The imitation-learning literature has already exposed the danger: behavioural cloning suffers from covariate shift because the learner visits states absent from the expert distribution, and causal confusion because the learner can attach the action to the wrong variable (Ross, Gordon & Bagnell, 2011; de Haan, Jayaraman & Levine, 2019).

The viability term $\mathcal{L}_{\mathrm{viab}}$ estimates whether predicted trajectories remain in $K_A$ or in $\operatorname{Viab}(K_A)$. In a soft version, it predicts a viability score alongside observations and penalises false safety. In a control-certified version, it includes barrier-function residuals such as

$$\dot b(q_t,a_t)+\kappa b(q_t) \geq 0,$$

for a safe set $K_A=\{q:b(q)\geq 0\}$. A model that reaches a goal by destroying the conditions for future action has not learned praxis for that goal. Viability is therefore not an external safety add-on. It is part of the representation target.

The composition term $\mathcal{L}_{\mathrm{comp}}$ encodes skill algebra. A skill is a morphism from structured preconditions to structured postconditions, in the sense of compositional systems where interfaces determine legal composition (Fong & Spivak, 2019):

$$m: A \rightarrow B.$$

Two skills compose when the postcondition of the first matches the precondition of the second. If $m:A\to B$ and $n:B\to C$, then the learned model should satisfy

$$
\sigma_{n\circ m} \approx \sigma_n \circ \sigma_m.
$$

This term prevents isolated competent fragments from masquerading as praxis. Picking up a plate and scrubbing a plate are locally successful skills; dishwashing competence requires their interface to match.

The active-sensing policy follows from the same mathematics. This connects P-JEPA to active perception, where sensing is controlled rather than passively received (Bajcsy, 1988), and to active inference, where action can reduce uncertainty under a generative model (Friston, 2010; Kaplan & Friston, 2018). P-JEPA does not identify obstruction with variational free energy. Its narrower claim is that local model disagreement can be made into a task-relevant epistemic action signal.

Let $R(\sigma)=\|d\sigma\|^2$ be representational contradiction. A useful probe is one that is expected to reduce contradiction while preserving viability:

$$
\alpha^\ast
=
\arg\max_{\alpha \in \mathcal{T}}
\left\{
\mathbb{E}\left[
R(\sigma_t)-R(\sigma_{t+k})
\mid do(\alpha)
\right]
- c(\alpha)
\right\},
$$

subject to $\Pr(q_{t:t+k}\in K_A)\geq 1-\delta$. P-JEPA therefore trains the representation and supplies a criterion for when the agent should touch, look, slow down, ask, grip more softly, or retreat.


## 7. Learning P-representations

The structure cannot be handed to the agent as notation. A trainable P-JEPA must infer its base space, local regions, local sections, restriction maps, and obstructions from action-history. The primitive datum is therefore

$$
(h_t,\tau,y_\tau),
$$

where $h_t$ is the action-observation history, $\tau=(a_t,\ldots,a_{t+k})$ is a short action test, and $y_\tau$ is the observed outcome, including ordinary observations, contact changes, task progress, and viability change. The atomic question is: given this embodied history, what happens under this intervention?

The base space is, ideally, a quotient of histories by action-consequence equivalence. Let $H$ be the space of histories available to the agent. Define

$$
h_1 \sim_P h_2
\quad \Longleftrightarrow \quad
\mathbb{P}(y\mid h_1,do(\tau)) \approx_\varepsilon
\mathbb{P}(y\mid h_2,do(\tau))
\text{ for relevant } \tau .
$$

Then

$$
\mathfrak{X}_A \approx H/\!\sim_P .
$$

This quotient is the central learning move. In practice the agent learns a finite approximation to it. Visually similar histories split when their action consequences differ; visually different histories merge when the same tests have the same consequences. A dry plate and a soapy plate are separated because lifting, rotating, and gripping have different outcome distributions. Two different-looking plates may merge when their grip and scrub affordances coincide.

Local regions $U_i$ are learned as domains on which one predictive model has low error. The agent assigns a new history to a region when a local model predicts the tested outcomes well; it marks an overlap when several local models predict well; it creates a new region when none do. Persistent structured error splits a region. Persistent action-equivalence and low overlap disagreement merge regions. In this way the cover of $\mathfrak{X}_A$ grows by prediction failure and simplifies by bisimulation-like equivalence. The base space is therefore not a geometric prior alone. It is a learned partition of action meaning.

Teaching and imitation enter at this level, but not as direct policy transfer. A demonstration supplies candidate successful sections. A correction localises obstruction. A warning marks a viability boundary. A verbal label can name a region. A recovery episode supplies an alternate section after failure. In P-JEPA, teaching helps the learner discover $\mathfrak{X}_A$, $U_i$, $\sigma_i$, $\rho_{ij}$, and $K_A$; it does not replace the learner's own intervention model.

A local section is a local predictive state:

$$
\sigma_i(h,\tau)
=
P(y_\tau \mid h,do(\tau), h\in U_i).
$$

This is the learned affordance model on $U_i$. A wet-grasp section may assign high probability to slip under fast lift and lower probability under slow lift. A dry-grasp section may assign the reverse pattern. The local section is therefore a map from tests to consequences; scene labels are secondary.

Restriction maps are learned on overlaps. If $h\in U_i\cap U_j$, then both $\sigma_i$ and $\sigma_j$ claim local authority over the same embodied situation. The maps

$$
\rho_{i,ij}: \mathcal{F}(U_i)\to\mathcal{F}(U_i\cap U_j),
\qquad
\rho_{j,ij}: \mathcal{F}(U_j)\to\mathcal{F}(U_i\cap U_j)
$$

are trained so that restricted predictions agree on shared tests and shared variables:

$$
\mathcal{L}_{\mathrm{glue}}
=
\sum_{i,j}
\sum_{h\in U_i\cap U_j}
\sum_{\tau}
D\left(
\rho_{i,ij}(\sigma_i)(h,\tau),
\rho_{j,ij}(\sigma_j)(h,\tau)
\right).
$$

The obstruction score is the residual disagreement:

$$
\mathcal{O}(\sigma)=\|d\sigma\|^2.
$$

High obstruction is epistemically useful. It tells the agent where its local models fail to cohere. A safe probe is chosen by expected obstruction reduction under viability:

$$
\tau^\ast
=
\arg\max_{\tau}
\mathbb{E}\left[
\mathcal{O}(\sigma_t)-\mathcal{O}(\sigma_{t+k})
\mid do(\tau)
\right],
$$

subject to $\Pr(q_{t:t+k}\in K_A)\geq 1-\delta$. If vision predicts stable pose, touch predicts incipient slip, and force predicts a dangerous grip, the agent should slow down, adjust force, add tactile sampling, or run a small tilt test. The probe is an action taken to repair the representation before continuing the task.

The learned cover has its own topology. From the cover $\{U_i\}$ form the nerve: an edge for every non-empty $U_i\cap U_j$, a triangle for every non-empty triple overlap, and so on. When the cover is a good approximation to the interaction space, cohomology on this nerve gives a computable summary of coherent competence and unresolved contradiction. $H^0$ records compatible components of action knowledge. $H^1$ records residual incompatibilities around loops in the learned cover. Higher obstructions record multi-region incompatibilities that pairwise checks miss.

The complete learning loop is therefore active predictive-state construction with sheaf consistency. Observe the current history. Assign it to local regions. Predict the outcomes of candidate tests. Compute overlap inconsistency. Act toward the task when obstruction is low. Probe safely when obstruction is high. Update local sections, restriction maps, viability estimates, and the cover itself. Repeat.


## 8. Why current JEPA starts elsewhere

Current JEPA work is optimized for scalable latent prediction. The working recipe is compact: mask image or video regions, encode visible context, predict target representations, scale a transformer, and evaluate the resulting embeddings. I-JEPA and V-JEPA show that this recipe can learn useful visual structure. V-JEPA 2 extends the path by pretraining on large-scale video, aligning to language for video question answering, and post-training an action-conditioned latent world model on a smaller robot-video corpus (Assran et al., 2025). V-JEPA 2.1 continues the same scaling path with denser predictive losses and stronger video representations (Mur-Labadia et al., 2026).

The P-JEPA problem is heavier. It asks the model to discover the interaction space, learn local regimes, learn local predictive-state sections, learn restriction maps, compute obstruction, and use obstruction to select interventions. This has no simple foundation-model recipe comparable to block masking and latent prediction over homogeneous patch tokens.

The difference is engineering as much as mathematical. Large AI systems are organised around differentiable architectures, enormous datasets, accelerator utilisation, and benchmark curves. Sheaf-valued praxis asks prior structural questions: what counts as a local context, what variables survive restriction, what makes two histories action-equivalent, what constitutes an obstruction, and which probes are viable. Those questions slow the scaling loop.

Robotics and control already use many of the pieces separately. Motion planning uses configuration-space topology. Safe control uses viability, reachability, and barrier functions. Contact-rich robotics uses hybrid systems. Geometric control handles body-specific controllability. Sensor integration and graph learning use sheaves. Imitation-learning theory identifies covariate shift and causal confusion. Predictive State Representations give action-conditioned state. The missing object is their integration into a trainable embodied architecture with benchmark pressure.

Recent work around JEPA still concentrates on representation quality and collapse. C-JEPA, for example, connects I-JEPA with VICReg-style variance, invariance, and covariance regularisation to address collapse and patch-representation limitations (Mo & Tong, 2024). This is the right problem for visual self-supervised learning. It is upstream of the local-to-global praxis problem.

The field starts with latent prediction because latent prediction scales. P-JEPA begins where scaling alone leaves a gap: local action models have to agree on their overlaps, and the agent must learn which intervention will repair the disagreement while preserving viability.


## 9. Hidden-regime simulation

The accompanying simulation tests the smallest version of the learning principle. All objects are visually plate-like. The hidden regime is one of four values: dry, soapy, cracked, or heavy. The agent has four direct task actions, `lift_fast`, `lift_slow`, `grip_hard`, and `two_contact_lift`, plus three safe probes, `shear_probe`, `tap_probe`, and `weigh_probe`. The regimes define local predictive sections: the same action has different success and unsafe-failure probabilities under each hidden regime. The evaluation is exact expectation over a uniform hidden-regime prior and stochastic probe evidence.

The experiment yields four findings. First, appearance is insufficient: visible-class and ungated mixture policies fail because they treat every object as the dry section. Second, action prediction improves over appearance: prior predictive policies and the PSR-only baseline choose the best average action. Third, action prediction alone does not repair representation: these baselines do not gather information when local sections disagree. Fourth, active probing improves the outcome because the agent can value information before committing to a task action.

The prior obstruction, computed as the posterior-weighted variance of local action-success sections, is

$$\|d\sigma\|^2 = 0.255.$$

At that point the visible class gives no regime information. Appearance-only and ungated mixture policies choose the dry-regime action `lift_fast`. The prior predictor, JEPA-style prior, and PSR-only baselines choose the prior-best direct action `two_contact_lift`. The pure sheaf policy chooses probes by expected obstruction reduction. The active PSR policy chooses probes by expected risk-adjusted value of information. The full P-JEPA stack uses obstruction as a coherence gate, but chooses probes and task actions by viability-aware value.

\begin{center}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Agent & Success & Unsafe & Probes & Obs. at action & Score \\
\midrule
Visible class & 0.450 & 0.473 & 0.000 & 0.255 & -0.495 \\
Prior predictor & 0.753 & 0.150 & 0.000 & 0.255 & 0.452 \\
PSR only & 0.753 & 0.150 & 0.000 & 0.255 & 0.452 \\
Active PSR probe & 0.853 & 0.081 & 1.987 & 0.165 & 0.651 \\
Ungated mixture & 0.450 & 0.473 & 0.000 & 0.255 & -0.495 \\
Entropy probe & 0.853 & 0.098 & 2.460 & 0.074 & 0.608 \\
Sheaf probe & 0.851 & 0.100 & 2.759 & 0.072 & 0.596 \\
P-JEPA stack & 0.853 & 0.081 & 1.987 & 0.165 & 0.651 \\
Oracle regime & 0.907 & 0.052 & 0.000 & 0.000 & 0.802 \\
\bottomrule
\end{tabular}
\end{center}

The table separates prediction, generic uncertainty reduction, obstruction reduction, and viability-aware intervention. The PSR-only baseline has action-conditioned predictions and selects the prior-best action, but it has no active information-gathering step. The mixture baseline has local regime models, but visible gating assigns every object to the dry section and leaves the overlap disagreement unused. The entropy baseline chooses probes by expected posterior-entropy reduction. It is a serious comparator, scoring $0.608$, and it beats pure obstruction reduction in the base suite because it probes slightly less. The pure sheaf policy reduces obstruction most strongly, lowering obstruction at action to $0.072$, but it over-probes relative to the risk-adjusted objective. The full stack stops earlier, leaves more residual obstruction ($0.165$), but obtains lower unsafe failure ($0.081$) and higher score ($0.651$). In this toy benchmark, the active PSR probe and the P-JEPA stack tie because exact Bayesian value of information already selects the same probes once local sections are known.

The hidden-regime breakdown shows where the gain appears. On the cracked regime, the prior-best `two_contact_lift` policy succeeds with probability $0.550$ and fails unsafely with probability $0.350$. The full stack probes, shifts toward `lift_slow`, and obtains success $0.811$ with unsafe failure $0.125$. On the heavy regime, the prior-best policy succeeds at $0.720$ with unsafe failure $0.180$; the full stack shifts toward `grip_hard`, succeeds at $0.851$, and fails unsafely at $0.100$. The soapy regime is the one case where the prior-best direct action is already the correct safe action, so probing mainly trades small success loss for lower regime uncertainty.

A benchmark sweep gives the boundary condition. With score

$$
S=\mathrm{success}-2\,\mathrm{unsafe}-\lambda_p\,\mathrm{probes},
$$

using $\lambda_p=0.02$ except in the costly-probe suite where $\lambda_p=0.04$, the full stack improves over the prior predictive baseline on the base suite ($0.651$ versus $0.452$), the held-out high-risk suite ($0.557$ versus $0.105$), the noisy-probe suite ($0.508$ versus $0.452$), the costly-probe suite ($0.500$ versus $0.452$), and the miscalibrated learned-section suite ($0.651$ versus $0.452$). It also beats entropy probing on every suite: by $0.043$, $0.064$, $0.032$, $0.110$, and $0.035$ score points respectively. The costly-probe suite exposes the difference between pure obstruction reduction and viability-aware probing: the sheaf probe scores $0.348$, while the P-JEPA stack scores $0.500$ by probing less and accepting some residual obstruction when further information is not worth its cost. The miscalibrated suite separates the agent's belief sections from the true world sections; the stack still improves because the learned model preserves enough action-consequence structure for safe probes to be valuable. The useful regime for P-JEPA is therefore not every hidden-state problem. It is the class of problems where information-gathering can be valued against the action failures it prevents.

The code also includes a representation-learning sanity check. Train and test contexts have shifted visual labels: the same visual code does not identify the same action regime across splits. A visual grouping learner therefore has test cluster purity $0.500$ and risk-adjusted score $0.282$. A single prior-average action model scores $0.453$. Clustering unlabeled contexts by action/probe fingerprints recovers the action regimes with purity $1.000$ and scores $0.802$, matching the oracle regime representation. This experiment is not a neural perception result; it is a controlled test of the quotient principle $h_1\sim_P h_2$, showing that downstream action learning should group situations by action consequences rather than by appearance when the two disagree.

A local video-representation surrogate makes the JEPA comparison executable without claiming a V-JEPA result. Each context is a short rendered video; a passive predictor is trained to predict a future frame from context frames, and its predicted target embedding is used as the learned representation. This passive model is competent at its own objective, with future-frame mean absolute error $0.017$. But under the train/test visual-style shift, that representation recovers action regimes poorly: cluster purity is $0.500$ and downstream score is $0.282$. The action-conditioned representation uses sampled intervention consequences instead of passive future frames; it reaches purity $1.000$ and score $0.802$, matching the oracle. Negative controls confirm that the gain depends on matched intervention evidence: replacing action evidence with random features drops score to $0.319$, and permuting test-time action evidence drops score to $0.085$. The result should be read narrowly. It shows that passive next-frame predictability can be the wrong representation criterion when the downstream task depends on hidden action consequences. It does not compare against V-JEPA, V-JEPA 2, or any video foundation model.

A load-bearing real-video smoke test checks this caution against actual video files. The code downloads the six official KTH sample AVI sequences, decodes them with `ffmpeg`, splits each sample video into temporal windows, and evaluates nearest-centroid classification using static appearance, passive next-frame, and temporal-motion descriptors (Schuldt, Laptev & Caputo, 2004). The result is not favourable to the P-JEPA story: static appearance scores $0.896$, the passive next-frame descriptor scores $0.805$, and temporal motion scores $0.623$. Because the split is within one sample video per class, background and actor appearance dominate. The result is therefore diagnostic rather than evidential. It shows that a non-surrogate video test can overturn the toy intuition, and that a serious video claim needs a full action-video or robot-video benchmark with subject/scene separation and, for P-JEPA specifically, intervention or action metadata. The repository therefore includes a manifest-based full-video protocol that rejects same-file train/test leakage, requires explicit group metadata by default, and can require action metadata, plus a KTH filename parser for full-dataset manifests. This is infrastructure for the next validity test rather than a new performance result; the six-file KTH sample correctly fails as a full benchmark.

A neural intervention-encoder benchmark removes the direct engineered fingerprint from the learner. Each training record contains a low-dimensional physical sensor observation, a test identity, and a sampled intervention outcome. Hidden regime labels are excluded from learner inputs and used only for evaluation. A small deterministic NumPy MLP predicts the outcome of each candidate test; the resulting predicted-test vector is then clustered and used to fit local action sections. Under the same shifted visual labels, the appearance-only encoder scores $0.282$ and the prior-average model scores $0.453$. The neural P-representation reaches purity $1.000$ and score $0.802$, matching the engineered fingerprint reference and the oracle score in this toy world. A direct sensor-only encoder also performs strongly, scoring $0.800$; the point of the neural P-representation result is not that clustering is always needed once sufficient sensors exist, but that the action-consequence fingerprint can be learned from intervention records rather than supplied as a table. This remains structured sensor learning, not pixel or tactile-stream perception.

The neural sample-efficiency sweep varies the number of sampled intervention repeats per context while holding the context stream fixed. At one repeat, the neural P-representation already scores $0.802$ with purity $1.000$, compared with $0.453$ for the prior baseline and $0.282$ for appearance grouping. Across $1,2,4,8,10$ repeats it stays within $0.000$ of the engineered fingerprint reference in risk-adjusted score; the minimum margin over the prior baseline is $0.345$ and the minimum margin over appearance grouping is $0.515$. The learned predicted-test mean absolute error decreases from $0.051$ at one repeat to $0.015$ at ten repeats. This should be read narrowly: sparse sampled intervention evidence is sufficient in this structured toy sensor world, not that P-JEPA is generally data-efficient in high-dimensional robotics.

A learned active-probing benchmark then makes the epistemic-action claim explicit. The initial structured sensor observation intentionally aliases dry with soapy and cracked with heavy, so the learner cannot resolve the safe action from sensors alone. The MLP receives sensor features, probe-evidence features, and test identities, and is trained from sampled intervention records; hidden regime labels are excluded from learner inputs. Acting immediately from the learned ambiguous representation scores $0.485$ with unsafe failure $0.143$. A learned entropy-probing policy scores $0.678$ with unsafe failure $0.078$ after $1.100$ probes on average. The learned value-aware active-probe policy scores $0.698$ with unsafe failure $0.069$ after $1.003$ probes on average, compared with the hidden-regime oracle score $0.803$. This result is narrow but important: under structured sensor aliasing, the learned predicted-test model can use safe probes to repair the representation before action, and value-aware probing improves the safety-efficiency frontier relative to generic entropy reduction.

The active-probing boundary sweep makes the limitation executable. In the aliased, informative setting, active probing scores $0.693$ versus $0.462$ for no probing and $0.686$ for entropy probing. When the initial sensors are made distinct, no-probe, entropy, and active policies all score $0.802$, and active probing uses zero probes. When probe likelihoods are moved toward chance, active probing scores only $0.490$ versus $0.485$ for no probing, and entropy probing falls to $0.448$. When probes remain informative but their cost is raised, active probing scores $0.647$ versus $0.485$ for no probing and $0.578$ for entropy probing while using $0.625$ probes on average rather than the full two-probe budget. The boundary condition is therefore explicit: P-JEPA-style active probing is useful when action-relevant hidden structure remains after ordinary sensing and when safe probes carry enough information to justify their cost.

A five-seed robustness sweep repeats the aliased, informative learned-active-probing run. Active probing beats no probing on every tested seed: the mean score margin is $0.213$, the minimum score margin is $0.196$, the mean unsafe-failure reduction is $0.073$, and the minimum unsafe-failure reduction is $0.068$. The active policy's mean score is $0.689$ with standard deviation $0.013$, and the mean gap to the hidden-regime oracle is $0.113$. The comparison with entropy probing is weaker and should be stated as such: active probing's mean score advantage over entropy is only $0.005$, and one of the five seeds favours entropy by $0.014$. The robust claim is therefore not that value-aware probing uniformly dominates entropy in this toy neural setting. It is that learned active probing robustly improves over acting from the aliased representation, robustly reduces unsafe failure, and is competitive with entropy while exposing a task-value criterion that can trade probe information against safety and cost.

The next validity test removes the hand-written structured sensor vector. A local pixel continuous-control benchmark renders $12\times 12$ observations of a 2D reaching task. The image aliases pairs of hidden action-dynamics regimes, and the task actions are continuous controller rollouts rather than discrete dish actions. The learner is still a small NumPy MLP, but its inputs are rendered pixels, probe evidence, and test identity. In this harder local setting, acting from pixels without probing scores $0.515$ with unsafe failure $0.105$. Entropy probing scores $0.559$ with unsafe failure $0.093$. Pixel active probing scores $0.572$ with unsafe failure $0.089$ after $0.176$ probes on average. The hidden-regime oracle scores $1.000$. The result is deliberately modest: it shows that the mechanism survives a first move from structured sensors to rendered pixels and continuous controller rollouts, while also showing large remaining headroom.

An online variant removes the offline clustering pass. Contexts arrive in an unlabeled stream, and the learner creates or updates a local regime whenever the next action-consequence fingerprint is too far from the existing cover. With the threshold fixed in advance, the online cover discovers four clusters, reaches test purity $1.000$, and scores $0.802$, again matching the oracle. Appearance-based online grouping keeps only two visual clusters and scores $0.282$. This is a minimal cover-construction result: the learner builds the local regimes incrementally, but the fingerprints are still engineered summaries of action tests rather than learned sensory encodings.

A synthetic scaling sweep asks whether this engineered representation mechanism immediately collapses as the number of hidden action regimes grows. The generator varies the number of hidden regimes over $4,8,16,32$ while keeping only two visual labels and shifting those labels between train and test. With eight action primitives and noisy action-consequence fingerprints, action-consequence grouping beats appearance grouping at every tested count by at least $0.599$ risk-adjusted score points and beats the prior-average section by at least $0.599$. Its minimum cluster purity over the sweep is $0.938$, and its maximum gap to the oracle regime representation is $0.001$. In the largest case, the learned action-consequence cover scores $0.803$ while appearance grouping and prior averaging both score $0.199$. This is not evidence of foundation-model scaling. It is a controlled stress test showing that the toy mechanism remains numerically coherent when the hidden regime count is increased in a synthetic action space.

A restriction-map ablation isolates the sheaf claim from active probing. The benchmark gives three local action sections for the same context: a vision section, a contact section, and a load section. Each section predicts action utilities, but the contact and load sections use different local action-coordinate frames. If the agent averages the sections as if their coordinates already agreed, overlap residual is $0.384$ and score is $0.572$. Fitting linear restriction maps from unlabeled overlap records reduces residual to $0.017$ and raises score to $0.796$. The hand-coded restriction maps score $0.795$, and the hidden-regime oracle scores $0.803$. This is the smallest executable version of the local-to-global claim: local experts are not enough when their interfaces are incompatible; the action interfaces have to be learned or aligned before aggregation and composition are meaningful. It is still not neural sheaf learning, because the local section vectors and overlap records are engineered.

A second sanity check tests skill composition. Each regime requires a first skill that establishes an intermediate postcondition and a second skill whose precondition matches that postcondition. The correct chains are `no_prep->fast_lift` for dry objects, `wipe->two_contact_lift` for soapy objects, `cushion->slow_lift` for cracked objects, and `brace->grip_hard` for heavy objects. Under the same shifted visual labels, appearance grouping scores $0.438$ and prior averaging scores $0.421$. Action-consequence grouping recovers purity $1.000$, selects all four intended chains, and scores $0.829$, matching the oracle representation. This is the smallest executable version of the claim that P-representations should support composition, not only one-step action choice.

The practical-use claim is now bundled into a single action-grounding challenge. The challenge reuses the controlled local benchmarks and requires all of the following to hold at once: passive video representation predicts future frames but fails action-regime transfer; action-consequence representation beats appearance grouping and prior averaging; a learned predicted-test vector transfers without hidden-label features; learned safe probes repair aliased observations and reduce unsafe failure; learned restriction maps align incompatible local sections; and action-grounded skill composition selects the intended chains. The current run passes all seven challenge checks. The key margins are $0.520$ for action-conditioned representation over passive video, $0.520$ for neural predicted-test representation over appearance, $0.213$ for active probing over no probing, $0.224$ for learned gluing over identity/no-glue aggregation, and $0.390$ for skill composition over appearance grouping. This is the most compact current answer to whether P-JEPA is useful: it is useful as an action-grounding benchmark and mechanism for representation repair under controlled hidden regimes.

The representative soapy trace illustrates the mechanism. Starting from the uniform prior, obstruction is $0.255$. The full stack chooses `shear_probe`, because the expected value of information exceeds the probe cost and risk. Positive shear evidence moves posterior mass to the soapy section ($0.870$) and lowers obstruction to $0.060$. The policy then chooses `two_contact_lift`, whose true-regime success probability is $0.900$ and unsafe-failure probability is $0.060$. The operative event is representation repair before task action.


## 10. Preliminary Meta-World adapter benchmark

The hidden-regime table is exact but deliberately small. To check whether the same mechanism can be instrumented in a continuous-control simulator, the codebase also includes a preliminary adapter for Meta-World, a simulated manipulation benchmark for multi-task and meta-reinforcement learning (Yu et al., 2020). The adapter uses the `reach-v3` task. It does not change the visible task specification; instead it wraps the action channel in a hidden regime sampled from four values: nominal, slippery, fragile, and heavy. The regime changes action scale, action noise, and an unsafe-action threshold. The policy observes the normal task state and a posterior over regimes, but not the true regime. Three safe probes provide noisy evidence about the hidden regime before the controller acts.

This is not a full Meta-World learning result. The controller is scripted for reaching, not trained from pixels or demonstrations. The point of the adapter is narrower: can obstruction-selected probes improve the safety-efficiency tradeoff once P-JEPA's hidden-regime mechanism is embedded in a real continuous-control environment?

The comparison uses a balanced 100-episode schedule over hidden regimes. The risk-adjusted score is

$$
S=\mathrm{success}-2\,\mathrm{unsafe}-0.02\,\mathrm{probes}.
$$

The fixed no-probe strategy uses the scripted reach controller without hidden-regime inference. The belief no-probe strategy gives the controller a posterior interface but no evidence, so it remains at the prior. Random one-probe runs one random probe before acting. Random exhaustive runs all three probes. Entropy probe chooses probes by expected posterior-entropy reduction. Obstruction probe chooses probes by expected obstruction reduction and stops when residual obstruction falls below the external threshold. Oracle receives the true hidden regime and gives the upper reference point.

\begin{center}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Strategy & Success & Unsafe & Probes & Obs. at action & Score \\
\midrule
Fixed no probe & 1.000 & 0.250 & 0.000 & 0.220 & 0.500 \\
Belief no probe & 1.000 & 0.250 & 0.000 & 0.220 & 0.500 \\
Random one probe & 1.000 & 0.210 & 1.000 & 0.178 & 0.560 \\
Random exhaustive & 1.000 & 0.040 & 3.000 & 0.089 & 0.860 \\
Entropy probe & 1.000 & 0.100 & 1.910 & 0.101 & 0.762 \\
Obstruction probe & 1.000 & 0.060 & 1.000 & 0.148 & 0.860 \\
Oracle & 1.000 & 0.000 & 0.000 & 0.000 & 1.000 \\
\bottomrule
\end{tabular}
\end{center}

The result is modest but informative. Obstruction-selected probing reduces unsafe events from $0.250$ to $0.060$ relative to the fixed no-probe controller. Against the same one-probe budget, it improves over random probing: unsafe rate falls from $0.210$ to $0.060$ and score rises from $0.560$ to $0.860$. It also beats entropy probing: unsafe rate falls from $0.100$ to $0.060$, score rises from $0.762$ to $0.860$, and mean probe count falls from $1.91$ to $1.00$. Exhaustive random probing reaches a slightly lower unsafe rate ($0.040$) only by using three probes per episode. Obstruction probing obtains the same risk-adjusted score with one probe on average. The adapter therefore supports the narrow external claim: the obstruction signal is useful as a probe-selection rule, not merely as a post-hoc diagnostic.

A second run replaces the hand-specified probe and local-section model used by the agent with estimates fit from wrapper experience. The training set contains $768$ sampled probe outcomes and $1024$ sampled action-effect records, balanced across the four hidden regimes. The fitting procedure estimates probe likelihoods and local action-effect section vectors from labelled simulator experience; the true wrapper is still used for evaluation. This is supervised regime-labelled fitting. The learned probe-likelihood mean absolute error is $0.026$ and the learned local-section mean absolute error is $0.015$.

\begin{center}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Strategy & Success & Unsafe & Probes & Obs. at action & Score \\
\midrule
Fixed no probe & 1.000 & 0.250 & 0.000 & 0.200 & 0.500 \\
Belief no probe & 1.000 & 0.250 & 0.000 & 0.200 & 0.500 \\
Random one probe & 1.000 & 0.210 & 1.000 & 0.165 & 0.560 \\
Random exhaustive & 1.000 & 0.070 & 3.000 & 0.082 & 0.800 \\
Entropy probe & 1.000 & 0.110 & 1.910 & 0.093 & 0.742 \\
Obstruction probe & 1.000 & 0.080 & 1.370 & 0.128 & 0.813 \\
Oracle & 1.000 & 0.000 & 0.000 & 0.000 & 1.000 \\
\bottomrule
\end{tabular}
\end{center}

The supervised learned model preserves the effect, but with a smaller margin than the hand-specified adapter. Obstruction probing reduces unsafe events from $0.250$ to $0.080$ relative to no probing, beats same-budget random probing by $0.253$ score points, beats entropy probing by $0.071$ score points, and slightly beats exhaustive random probing by $0.013$ score points while using $1.37$ probes rather than $3.0$. This is the first point at which the code no longer merely consumes the hand-specified local sections: it estimates the probe model and the local action-effect section vectors from interaction data, then uses the learned obstruction to choose probes.

A third run removes regime labels from fitting and removes the balanced-by-regime training schedule. The learner receives $160$ unlabeled context fingerprints sampled from the prior stream; in the deterministic run used here, the stream contains $41$ nominal, $47$ slippery, $39$ fragile, and $33$ heavy contexts. Each fingerprint contains repeated probe outcomes and action-effect statistics, but not the hidden regime name. A simple $k$-means model clusters these fingerprints by action consequences and probe profiles, then fits one local section per discovered cluster. Hidden labels are used only after fitting to report diagnostic purity and to align cluster names for evaluation. The discovered clusters have purity $1.000$, probe-likelihood mean absolute error $0.0092$, and local-section mean absolute error $0.0236$.

\begin{center}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Strategy & Success & Unsafe & Probes & Obs. at action & Score \\
\midrule
Fixed no probe & 1.000 & 0.250 & 0.000 & 0.175 & 0.500 \\
Belief no probe & 1.000 & 0.250 & 0.000 & 0.175 & 0.500 \\
Random one probe & 1.000 & 0.210 & 1.000 & 0.140 & 0.560 \\
Random exhaustive & 1.000 & 0.040 & 3.000 & 0.072 & 0.860 \\
Entropy probe & 1.000 & 0.100 & 2.120 & 0.078 & 0.758 \\
Obstruction probe & 1.000 & 0.060 & 1.000 & 0.111 & 0.860 \\
Oracle & 1.000 & 0.000 & 0.000 & 0.000 & 1.000 \\
\bottomrule
\end{tabular}
\end{center}

The stream-unsupervised result is an important step beyond supervised fitting. The local regimes are not handed to the learner, and the training stream is not manually balanced by regime. The regimes are recovered from action/probe fingerprints sampled from the prior. The discovered obstruction then reproduces the hand-specified probe-efficiency result: it beats same-budget random probing, beats entropy probing by $0.102$ score points, and matches exhaustive random probing's score with one probe instead of three.

A fourth run removes the precomputed fingerprint interface. The learner receives a prior-sampled stream of raw event records: probe outcomes, raw actions, transformed actions, and unsafe-action indicators. Hidden-regime labels are kept outside the learner-facing record stream and used only for diagnostic counts and purity after fitting. The deterministic run contains $160$ contexts and $17{,}920$ raw records: $7{,}680$ probe records and $10{,}240$ action records. The context stream contains $38$ nominal, $43$ slippery, $38$ fragile, and $41$ heavy contexts. The learner derives context fingerprints from those records, clusters them by action consequences, and fits local sections as before. The discovered clusters again have purity $1.000$; probe-likelihood mean absolute error is $0.0094$ and local-section mean absolute error is $0.0249$.

\begin{center}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Strategy & Success & Unsafe & Probes & Obs. at action & Score \\
\midrule
Fixed no probe & 1.000 & 0.250 & 0.000 & 0.174 & 0.500 \\
Belief no probe & 1.000 & 0.250 & 0.000 & 0.174 & 0.500 \\
Random one probe & 1.000 & 0.210 & 1.000 & 0.137 & 0.560 \\
Random exhaustive & 1.000 & 0.060 & 3.000 & 0.072 & 0.820 \\
Entropy probe & 1.000 & 0.100 & 1.680 & 0.090 & 0.766 \\
Obstruction probe & 1.000 & 0.060 & 1.000 & 0.108 & 0.860 \\
Oracle & 1.000 & 0.000 & 0.000 & 0.000 & 1.000 \\
\bottomrule
\end{tabular}
\end{center}

The raw-record result is the strongest current implementation result. The hidden regimes are not labelled, the stream is not balanced by regime, and the learner is not handed precomputed local-section parameters. It still recovers the same action-consequence regimes, reduces unsafe events from $0.250$ to $0.060$ relative to no probing, beats same-budget random probing by $0.300$ score points, beats entropy probing by $0.094$ score points, and beats exhaustive random probing by $0.040$ score points while using one probe instead of three. The gain comes from probe selection, not from a better controller: all non-oracle strategies use the same scripted reach controller after inference.

This also clarifies what the result does not prove. It does not show that P-JEPA learns a controller in Meta-World. It does not show visual perception, multi-task transfer, or neural sheaf learning. It shows that the executable mechanism from the hidden-regime world can be embedded in a standard continuous-control simulator and can improve the probe-efficiency frontier under hidden dynamics, including when the agent's probe and local-section model is estimated from labelled wrapper experience, discovered by clustering unlabeled prior-sampled fingerprints, or derived from raw unlabeled probe/action records.


## 11. Formal verification interface

Kona and Aleph are not direct embodied-control baselines for the present simulation. Public descriptions frame Kona as an energy-based reasoning or constraint layer for deciding what states or actions are valid, safe, and permissible, and Aleph as an orchestration layer for formal verification that produces machine-checkable proofs (Logical Intelligence, 2026a; Logical Intelligence, 2026b). That makes them relevant to a different part of the P-JEPA stack: not learning the P-representation, but certifying properties of the learned policy interface once it has been exported into a finite or formal model.

The repository therefore adds a verification-interface benchmark rather than a claimed Kona/Aleph run. Each selected policy is exported as a finite contract over the configured hidden-regime suites. The current contract requires expected unsafe failure at most $0.13$, worst hidden-regime branch unsafe failure at most $0.20$, risk-adjusted score at least $0.49$, residual obstruction at action at most $0.22$, and mean probe count at most $2.50$. These thresholds are engineering targets for the local benchmark, not universal constants. A local exhaustive checker evaluates the finite state/action model and emits counterexamples when a requirement fails.

\begin{center}
\small
\begin{tabular}{@{}lrr@{}}
\toprule
Strategy & Suite contracts passed & Counterexamples \\
\midrule
Prior predictor & $0/5$ & $19$ \\
Entropy probe & $1/5$ & $8$ \\
P-JEPA stack & $5/5$ & $0$ \\
Oracle regime & $5/5$ & $0$ \\
\bottomrule
\end{tabular}
\end{center}

This benchmark is useful because it gives a precise interface between representation learning and certification. P-JEPA supplies a policy whose decisions depend on local action models, obstruction, probes, and viability. A proof or constraint backend can then be asked a narrower question: does this exported policy satisfy a stated safety or composition contract on the finite model, or can it produce a counterexample? The current implementation answers that question only with the local checker. It explicitly records that no external Kona, Aleph, Lean, or proprietary backend was executed.


## 12. Limits

P-JEPA remains a formal proposal with controlled numerical demonstrations. The exact simulation establishes a learning principle in a small hidden-regime world: local action models can disagree measurably, and the disagreement can drive safe probes that improve transfer. The representation-learning benchmark establishes the narrower quotient claim that grouping by action consequences can support downstream action choice when appearance is unstable. The video-representation benchmark is a local passive-prediction surrogate; it does not compare against V-JEPA, V-JEPA 2, or any video foundation model. The KTH sample benchmark uses real video but remains only a smoke test; the video and robot manifest protocols are guardrails for future full-dataset experiments, not results. The evidence-level guard is a further claim-boundary check: it classifies each verifier and prevents protocols, diagnostic negatives, and local surrogates from being counted as broad performance evidence. The action-grounding challenge bundles the current practical-use evidence into one controlled harness, but it inherits the limits of its component benchmarks and is not a robotics or foundation-model result. The neural benchmarks learn predicted-test vectors from sampled intervention records over low-dimensional physical sensor features; they are not tactile-stream learning or end-to-end robot learning, and the sample-efficiency sweep is a sparse-evidence toy result rather than a general data-efficiency claim. The learned active-probing benchmark still uses structured sensor and probe-evidence features plus an exact evidence-tree evaluator; it is evidence for learned value-aware probing under sensor aliasing, not for open-world robot experimentation. Its boundary sweep confirms the expected failure modes: if sensors already identify the regime, probing is unnecessary; if probes are weak, the gain nearly vanishes. Its seed sweep supports the no-probe and unsafe-failure claims across tested deterministic seeds, but it also weakens any broad claim against entropy probing: value-aware probing is only slightly better on average and can lose to entropy on an individual seed. The pixel continuous-control benchmark is the first local move toward learned perception and harder control, but it still uses tiny rendered images, a small MLP, finite controller templates, and simulated 2D dynamics. It is not MuJoCo-scale robot learning, and the oracle gap remains large. The formal contract-interface benchmark exports finite contracts and checks them locally; it is not a Kona, Aleph, Lean, or external formal-verification result. The online-cover benchmark shows incremental construction of such regimes from an unlabeled stream, but with engineered action-consequence fingerprints. The synthetic scaling benchmark increases hidden regime count under the same engineered-fingerprint assumption; it is not evidence of high-dimensional neural scaling. The gluing ablation learns linear restriction maps over engineered local section vectors; it is a controlled interface-alignment test, not an end-to-end learned sheaf. The skill-composition benchmark adds a minimal precondition/postcondition chain but still uses engineered skill tables. The Meta-World adapter establishes a second, weaker point: the same probe-selection mechanism can be placed around a continuous-control simulator and can improve probe efficiency under hidden action regimes. None of these results is a full learned robotics benchmark. The exact simulation uses configured local sections and exact expectation; one suite separates the agent's noisy belief sections from the true world sections, but the cover, probes, and regime vocabulary remain hand-specified. The representation, online-cover, scaling, gluing, and composition benchmarks use engineered action/probe fingerprints or local section vectors rather than learned perception. The Meta-World adapter uses a scripted reach controller and hand-defined hidden regimes. The supervised learned external run estimates probe likelihoods and local section vectors from labelled data. The stream-unsupervised external run discovers local regimes from unlabeled, prior-sampled fingerprints. The raw-record external run derives those fingerprints from unlabeled probe/action event records, but the event channels themselves are still engineered: the code supplies probe identities, raw/transformed action vectors, and unsafe-action indicators. It does not discover regimes online from tactile streams, language, or uncontrolled robot logs. The results also do not isolate a uniquely sheaf-theoretic advantage in the strong sense: in the exact benchmark, active PSR value-of-information and the P-JEPA stack choose the same probes; in the hand-specified and unsupervised external adapters, exhaustive random probing can buy comparable risk-adjusted score by spending more probes. The sheaf machinery contributes an explicit coherence diagnostic and a path to learned local-to-global consistency, but a harder benchmark is still needed where restriction maps, cover membership, and local sections are learned online and can be wrong together. The present result should therefore be read as evidence for the full stack of intervention, viability, active probing, learned predicted-test representation, action-grounded representation, formal contract export, online cover construction, synthetic regime-count scaling, restriction-map gluing, skill composition, and local-model consistency, not as proof that cohomology alone improves control. It does not establish scalability to high-dimensional robotics, real learned perception, open-world manipulation, or internet-scale video pretraining.

The construction also leaves several choices underdetermined. The interaction space $\mathfrak{X}_A$ may be hand-engineered, learned, or hybrid. The target category $\mathbf{PredControl}$ may be vector-valued, distribution-valued, controller-valued, or symbolic. The sheaf may be explicit, as in cellular sheaf models over graphs, or implicit, as a consistency objective over learned local experts. The viability set may be physically certified or statistically estimated. Each choice changes the scientific status of the resulting system.

There is a risk of decorative mathematics. Sheaves, cohomology, viability kernels, and categories earn their place only when they constrain training, diagnosis, or action. A paper diagram that labels every context a stalk has added vocabulary rather than competence. The operative tests are concrete: the model should identify contradictions on overlaps, choose probes that reduce them, preserve viability under intervention, and compose skills across regime transitions.


## 13. Conclusion

JEPA supplies a valuable predictive layer: learn representations by predicting representations. P-JEPA asks what that layer must become when the agent has a body, acts through interventions, crosses contact regimes, composes skills, and must preserve the conditions of further action. The answer is a sheaf-valued predictive state over a stratified interaction space, constrained by viability and reachability.

The central object is

$$
P(A) =
\left(
\mathfrak{X}_A,
\mathcal{F}_{A},
K_A,
\mathcal{R}_A,
\mathcal{C}_A,
\Pi_A
\right),
$$

where $\mathfrak{X}_A$ is the agent's interaction space, $\mathcal{F}_A$ is its affordance sheaf, $K_A$ its viability constraints, $\mathcal{R}_A$ its reachable sets, $\mathcal{C}_A$ its category of skills, and $\Pi_A$ its available policies or controllers. For a goal family $G$, the agent has praxis when some section

$$\sigma \in H^0(\mathfrak{X}_A,\mathcal{F}_A)$$

and some controller $\pi \in \Pi_A$ carry the trajectory into $G$ while remaining inside the viability kernel. When $d\sigma$ is large, the failure is localisable: the agent has local models that do not yet cohere on their overlaps. When the corresponding cohomology class is non-trivial, the incompatibility cannot be removed by local correction alone. That obstruction is a training target, a diagnostic, and a reason to act.

The executable benchmarks give the smallest current version of the claim. In the exact hidden-regime simulation, the sheaf policy begins with the same prior as the baselines and succeeds because it treats section disagreement as a quantity to reduce before acting. In the Meta-World adapter, the same obstruction signal improves probe selection under hidden action regimes. That is the operational distinction between a latent predictor and a praxis learner.

The shortest formula is therefore:

$$
P = \text{viable reachability through a sheaf of action-conditioned predictive states}.
$$

Ordinary JEPA predicts what the world will represent. P-JEPA learns the cover of the agent's action world, the local sections valid on that cover, the restrictions that make neighbouring sections comparable, and the obstructions that direct further experiment. It predicts what the agent can do without losing the world in which doing remains possible.


## References

Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., & Tabuada, P. (2019). Control barrier functions: Theory and applications. *2019 18th European Control Conference*.

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. arXiv:2301.08243.

Assran, M., et al. (2025). V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv:2506.09985.

Aubin, J.-P. (1991). *Viability Theory*. Birkhauser.

Ayzenberg, A., Gebhart, T., Magai, G., & Solomadin, G. (2025). Sheaf theory: from deep geometry to deep learning. arXiv:2502.15476.

Bajcsy, R. (1988). Active perception. *Proceedings of the IEEE*, 76(8), 966-1005.

Bansal, S., Chen, M., Herbert, S. L., & Tomlin, C. J. (2017). Hamilton-Jacobi reachability: A brief overview and recent advances. arXiv:1709.07523.

Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., & Ballas, N. (2024). Revisiting feature prediction for learning visual representations from video. arXiv:2404.08471.

Bodnar, C., Di Giovanni, F., Chamberlain, B. P., Lio, P., & Bronstein, M. M. (2022). Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in GNNs. arXiv:2202.04579.

Boots, B., Siddiqi, S. M., & Gordon, G. J. (2011). Closing the learning-planning loop with predictive state representations. *The International Journal of Robotics Research*, 30(7), 954-966.

Bullo, F., & Lewis, A. D. (2004). *Geometric Control of Mechanical Systems*. Springer.

Curry, J. (2014). *Sheaves, Cosheaves and Applications*. PhD thesis, University of Pennsylvania.

de Haan, P., Jayaraman, D., & Levine, S. (2019). Causal confusion in imitation learning. *NeurIPS 2019*.

Ferns, N., Panangaden, P., & Precup, D. (2011). Bisimulation metrics for continuous Markov decision processes. *SIAM Journal on Computing*, 40(6), 1662-1714.

Fong, B., & Spivak, D. I. (2019). *Seven Sketches in Compositionality: An Invitation to Applied Category Theory*. Cambridge University Press.

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11, 127-138.

Goebel, R., Sanfelice, R. G., & Teel, A. R. (2012). *Hybrid Dynamical Systems: Modeling, Stability, and Robustness*. Princeton University Press.

Hansen, J., & Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. *Journal of Applied and Computational Topology*, 3, 315-358.

LeCun, Y. (2022). A path towards autonomous machine intelligence. Version 0.9.2.

Kaplan, R., & Friston, K. J. (2018). Planning and navigation as active inference. *Biological Cybernetics*, 112, 323-343.

Littman, M. L., & Sutton, R. S. (2001). Predictive representations of state. *Advances in Neural Information Processing Systems 14*.

Logical Intelligence. (2026a). Kona: Energy-based models for AI reasoning. https://logicalintelligence.com/kona-ebms-energy-based-models

Logical Intelligence. (2026b). Aleph: The formal verification coding AI agent. https://logicalintelligence.com/aleph-coding-ai/

Mo, S., & Tong, S. (2024). Connecting joint-embedding predictive architecture with contrastive self-supervised learning. *NeurIPS 2024*.

Mur-Labadia, L., Muckley, M., Bar, A., Assran, M., Sinha, K., Rabbat, M., LeCun, Y., Ballas, N., & Bardes, A. (2026). V-JEPA 2.1: Unlocking dense features in video self-supervised learning. arXiv:2603.14482.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. 2nd edition. Cambridge University Press.

Robinson, M. (2017). Sheaves are the canonical data structure for sensor integration. *Information Fusion*, 36, 208-224.

Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *AISTATS 2011*.

Schuldt, C., Laptev, I., & Caputo, B. (2004). Recognizing human actions: A local SVM approach. *Proceedings of the 17th International Conference on Pattern Recognition*.

Singh, S., Littman, M. L., Jong, N. K., Pardoe, D., & Stone, P. (2003). Learning predictive state representations. *ICML 2003*.

Yu, T., Quillen, D., He, Z., Julian, R., Hausman, K., Finn, C., & Levine, S. (2020). Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning. *Proceedings of the Conference on Robot Learning*, PMLR 100, 1094-1100.

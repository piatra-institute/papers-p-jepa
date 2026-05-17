# Architecture

P-JEPA has two linked artifacts: a scientific paper defining P-representations and P-JEPA, and a simulation suite that tests the smallest executable version of the mechanism.

The implementation is intentionally modest. It does not claim full robotics or foundation-model learning. It tests whether action-conditioned local models, obstruction, viability, and active probing improve decisions under hidden physical regimes.

## Conceptual Stack

The paper's core object is a P-representation:

```text
history -> action-conditioned predictive state -> local model cover
        -> gluing/obstruction check -> probe or task action
```

The main components are:

- Predictive state: a situation is represented by predicted outcomes of action tests, not only by appearance.
- Local sections: each hidden regime has a local action model.
- Obstruction: disagreement among local action models under the current posterior, implemented as a weighted variance of local section predictions.
- Active probing: safe tests are chosen to reduce obstruction or improve risk-adjusted action value.
- Viability: unsafe outcomes and probe costs are part of the decision criterion.
- Skill/action choice: task action is selected after representation repair.

## Code Layers

```text
paper/
  PAPER.md                scientific text
  PAPER.pdf               generated PDF artifact, ignored by git

simulation/
  pjepa_sim/
    core/                 exact hidden-regime world and original agents
    benchmark/            suite evaluator, suite configs, and P-JEPA stack
      configs/            benchmark suite definitions
    external/             Meta-World adapter and learned estimators
    representation/       action-grounded representation benchmark
      clustering.py       shared deterministic clustering helpers
    verification/         executable claim checks
    cli/                  command implementations
    paths.py              repository-local output/config paths
  pyproject.toml          uv project metadata
  uv.lock                 locked Python environment
```

Key implementation modules:

```text
pjepa_sim/core/dishworld.py
pjepa_sim/core/agents.py
pjepa_sim/benchmark/suites.py
pjepa_sim/external/
    metaworld_hidden_regime.py  hidden-regime wrapper and strategies
    learned_metaworld.py        learned/local-section estimators
pjepa_sim/verification/
    *_claims.py                 executable claims
    reporting.py                shared verifier reporting helpers
```

## Exact Hidden-Regime World

`simulation/pjepa_sim/core/dishworld.py` defines the small controlled world: all objects look like plates, hidden regimes are `dry`, `soapy`, `cracked`, and `heavy`, direct actions are `lift_fast`, `lift_slow`, `grip_hard`, and `two_contact_lift`, and probes are `shear_probe`, `tap_probe`, and `weigh_probe`.

Each regime defines success and unsafe-failure probabilities for each action. Probes produce noisy evidence about the hidden regime.

The exact evaluator enumerates stochastic evidence rather than sampling it. That makes the main toy result deterministic and easy to audit.

## Benchmark Suites

`simulation/pjepa_sim/benchmark/suites.py` generalizes the exact world into configurable suites:

- `hidden_regime_v0`: base hidden-regime manipulation.
- `heldout_regime_v0`: evaluation prior shifts toward high-risk regimes.
- `noisy_probe_v0`: probe evidence is weaker.
- `costly_probe_v0`: probes are more expensive.
- `miscalibrated_sections_v0`: the policy acts from noisy learned local sections while evaluation uses the true world model.

The key agents are:

- `model_based_prior`: chooses the best action under the prior.
- `entropy_probe`: chooses probes by expected posterior-entropy reduction.
- `sheaf_probe`: chooses probes by expected obstruction reduction.
- `active_psr_probe`: chooses probes by exact value of information.
- `p_jepa_stack`: uses obstruction as a coherence gate and viability-aware value for probe/action choice.
- `oracle_hidden_regime`: upper reference point with access to the true regime.

## Representation Benchmark

`simulation/pjepa_sim/representation/learning.py` tests the quotient principle separately from active probing. Train and test contexts deliberately shift visual labels, so appearance is not a stable action representation. The action-consequence learner clusters unlabeled contexts by action/probe fingerprints, fits local action sections per cluster, and then uses those sections for downstream action choice. This is still engineered fingerprint learning, not perception, but it tests whether representations learned from action consequences are more useful than representations learned from appearance.

`simulation/pjepa_sim/representation/online.py` removes the offline clustering pass. Contexts arrive in an unlabeled stream, and the learner creates or updates a local model when the next action-consequence fingerprint is outside the current cover. This tests incremental cover construction, still from engineered action-test summaries rather than sensory encoders.

`simulation/pjepa_sim/representation/scaling.py` varies the number of synthetic hidden action regimes from 4 to 32. The action-consequence learner is compared with prior averaging, appearance grouping, and an oracle regime model. This is a controlled scaling sanity check for the representation mechanism, not evidence of foundation-model or robot scaling.

`simulation/pjepa_sim/representation/gluing.py` isolates the local-to-global part. Multiple local action sections observe the same context through incompatible action-coordinate frames. The no-glue baseline averages them as if their coordinates already agreed; the learned-glue model fits restriction maps from unlabeled overlap records and then aggregates the aligned sections. This is an explicit restriction-map ablation, not neural sheaf learning.

`simulation/pjepa_sim/representation/composition.py` extends the same idea to a two-step task. The first skill creates an intermediate postcondition and the second skill must be valid for that postcondition. The benchmark tests whether the learned action-consequence representation supports skill-chain selection rather than only one-step action choice.

## External Adapter

`simulation/pjepa_sim/external/metaworld_hidden_regime.py` wraps Meta-World `reach-v3` with hidden action regimes. The environment state remains the normal task state, but the action channel is transformed by an unobserved regime: `nominal`, `slippery`, `fragile`, or `heavy`.

The adapter adds hidden-regime posterior updates, safe probe events, obstruction logging, unsafe-action tracking, and scripted reach controller variants.

This is an adapter benchmark, not a trained Meta-World policy result.

## Learned Local Sections

`simulation/pjepa_sim/external/learned_metaworld.py` progressively removes hand-specified knowledge from the agent:

1. Supervised learned model: estimates probe likelihoods and local action section parameters from labelled wrapper experience.
2. Balanced unsupervised model: clusters unlabeled context fingerprints.
3. Stream-unsupervised model: samples contexts from the prior stream instead of balancing by regime.
4. Raw-record model: starts from unlabeled probe/action event records, derives fingerprints internally, then clusters local regimes.

Hidden labels are used for diagnostics and evaluation, not as learner features in the unsupervised or raw-record paths.

## Evidence Flow

```text
raw events / fingerprints / exact tables
  -> local section estimates
  -> posterior over regimes
  -> obstruction or entropy probe selection
  -> task action
  -> success, unsafe, probes, score
  -> verifier margins
  -> paper claims
```

The paper should only make claims that are backed by verifier scripts or explicitly marked as formal proposal / future work.

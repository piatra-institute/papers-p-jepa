# Scientific Claims

This project should make narrow, executable claims. It should not claim that P-JEPA solves robotics, learns perception, scales to foundation models, or establishes cohomology as independently superior to all active-learning objectives.

## Main Demonstrated Claim

In hidden-regime manipulation settings where visually identical situations have different action consequences, an agent can improve safety or probe efficiency by representing local action-conditioned predictive models, measuring local-model disagreement as obstruction, choosing safe probes to reduce the relevant disagreement, selecting task actions after the posterior has been repaired, and valuing information against unsafe outcomes and probe costs.

## Exact Suite Evidence

`simulation/pjepa_sim/benchmark/suites.py` evaluates exact evidence trees over the hidden regime world.

Current suite-level executable checks:

- `p_jepa_stack` beats the prior predictive baseline on risk-adjusted score.
- `p_jepa_stack` beats pure obstruction reduction on risk-adjusted score.
- `p_jepa_stack` beats posterior-entropy probing on risk-adjusted score.
- `p_jepa_stack` reduces unsafe failure relative to the prior baseline.
- `p_jepa_stack` uses fewer probes than pure obstruction reduction.
- `p_jepa_stack` fixes the costly-probe boundary where pure obstruction reduction is penalized.
- `p_jepa_stack` still improves in the miscalibrated-section suite, where the policy's belief model differs from the true world model.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.benchmark_claims
```

## Representation-Learning Evidence

`simulation/pjepa_sim/representation/learning.py` tests whether a representation learned from action/probe fingerprints supports downstream action choice when visual cues shift between train and test.

Current executable checks:

- Action-consequence grouping beats appearance grouping on risk-adjusted score.
- Action-consequence grouping beats the prior-average action model.
- Action-consequence clusters recover hidden action regimes with high purity.
- Action-consequence grouping approaches the oracle regime score.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.representation_claims
```

## Online Cover-Construction Evidence

`simulation/pjepa_sim/representation/online.py` tests whether the action-consequence cover can be constructed incrementally from an unlabeled stream instead of fit by one offline clustering pass.

Current executable checks:

- The online learner discovers the four action regimes from engineered action/probe fingerprints.
- The discovered clusters recover hidden action regimes with high purity.
- Online action-consequence cover construction beats appearance-based online grouping.
- Online action-consequence cover construction beats the prior-average action model.
- Online action-consequence cover construction approaches the oracle regime score.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.online_claims
```

## Synthetic Scaling Evidence

`simulation/pjepa_sim/representation/scaling.py` tests a controlled synthetic sweep over 4, 8, 16, and 32 hidden action regimes. Visual labels remain low-cardinality and shift between train and test; action-consequence fingerprints remain the stable basis for local sections.

Current executable checks:

- Action-consequence grouping beats appearance grouping at every tested regime count.
- Action-consequence grouping beats the prior-average action model at every tested regime count.
- The learned action-consequence cover keeps high regime purity across the sweep.
- Action-consequence grouping approaches the oracle regime score across the sweep.
- The 32-regime case remains useful rather than collapsing.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.scaling_claims
```

## Restriction-Map Gluing Evidence

`simulation/pjepa_sim/representation/gluing.py` tests the local-to-global consistency claim directly. The benchmark gives the agent multiple local action sections over the same contexts, but the sections use incompatible local action-coordinate frames. The learned-glue model fits restriction maps from unlabeled overlap records before aggregating predictions.

Current executable checks:

- Learned restriction maps reduce overlap residual relative to identity/no-glue aggregation.
- Learned restriction maps improve risk-adjusted action score relative to identity/no-glue aggregation.
- Learned restriction maps improve over the noisy reference section alone.
- Learned restriction maps approach hand-coded oracle restriction maps.
- Learned restriction maps recover most of the hidden-regime oracle action value in this toy setup.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.gluing_claims
```

## Skill-Composition Evidence

`simulation/pjepa_sim/representation/composition.py` tests whether action-grounded representations support two-step skill chains. Each regime requires a preparation skill that creates an intermediate postcondition and a finishing skill valid for that postcondition.

Current executable checks:

- Action-consequence grouping beats appearance grouping on composition score.
- Action-consequence grouping beats the prior-average chain model.
- Action-consequence clusters recover hidden action regimes with high purity.
- Action-consequence grouping approaches the oracle regime composition score.
- The learned representation selects the expected prepare/finish chains for all regimes.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.composition_claims
```

## External Adapter Evidence

The Meta-World adapter embeds the hidden-regime mechanism in `reach-v3`. It uses a scripted reach controller, not a trained policy.

The external checks test whether obstruction-selected probing improves the probe-efficiency frontier under hidden action dynamics.

Current executable comparisons include no-probe baseline, same-budget random probing, exhaustive random probing, posterior-entropy probing, obstruction probing, and oracle hidden-regime access.

Verifier commands:

```bash
cd simulation
uv run python -m pjepa_sim.verification.external_claims
uv run python -m pjepa_sim.verification.learned_external_claims
uv run python -m pjepa_sim.verification.unsupervised_external_claims
uv run python -m pjepa_sim.verification.stream_external_claims
uv run python -m pjepa_sim.verification.raw_record_external_claims
```

## Learned-Model Ladder

The external adapter has a ladder of increasingly less hand-specified models:

1. Hand-specified probe and local-section model.
2. Supervised labelled fitting from wrapper experience.
3. Balanced unsupervised clustering from context fingerprints.
4. Prior-stream unsupervised clustering.
5. Raw-record learner deriving fingerprints from unlabeled probe/action records.

The strongest current implementation result is the raw-record run. It starts from unlabeled event records, recovers local action-consequence regimes, and then uses learned obstruction to select probes.

## What Is Not Demonstrated

The current implementation does not demonstrate learning a robot controller, visual perception from pixels, tactile representation learning, language grounding, online regime discovery from uncontrolled logs, multi-task robot transfer, neural sheaf learning, a uniquely sheaf-theoretic advantage isolated from intervention, viability, and value-of-information effects, or scalability to internet video or foundation-model pretraining. The synthetic scaling benchmark is a controlled regime-count sweep over engineered action-consequence fingerprints, not evidence of high-dimensional neural scaling. The gluing ablation learns linear restriction maps over engineered local section vectors; it is not an end-to-end neural sheaf. The representation, online cover-construction, and composition benchmarks use engineered action/probe fingerprints and skill tables rather than learned sensory encoders or learned options.

These limits are not defects to hide. They define the correct scientific status: formal proposal plus controlled executable demonstrations.

## Claim Hygiene

- Do not cite internal PIATRA papers as evidence.
- Do not turn a toy benchmark into a robotics claim.
- Do not report generated numbers without a command that regenerates them.
- Do not compare against weak baselines only; keep entropy and random probing in the comparison.
- Keep the distinction between pure obstruction reduction and the full viability-aware P-JEPA stack explicit.

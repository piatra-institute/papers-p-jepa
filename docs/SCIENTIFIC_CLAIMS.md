# Scientific Claims

This project should make narrow, executable claims. It should not claim that P-JEPA solves robotics, learns perception, scales to foundation models, or establishes cohomology as independently superior to all active-learning objectives.

For an auditable local summary, run `uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download` once from `simulation/`, then run `uv run python -m pjepa_sim.cli.verify_all`. It writes `output/CLAIMS_SUMMARY.md`, which lists every local claim, its verifier JSON, observed value, threshold, and limitation. It also writes `output/EVIDENCE_MATRIX.md`, which separates demonstrated local mechanisms, learned structured-sensor results, synthetic scaling, local surrogates, diagnostic negatives, and protocol-only infrastructure. The KTH sample real-video check is included in this command; optional Meta-World checks are excluded because they require external simulator dependencies.

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

## Neural Intervention Encoder Evidence

`simulation/pjepa_sim/representation/neural.py` tests whether the action-consequence fingerprint can be learned from intervention records rather than supplied directly. The learner receives low-dimensional physical sensor observations and test identities, then predicts sampled intervention outcomes. Hidden regime labels are excluded from learner inputs and used only for diagnostics and evaluation.

Current executable checks:

- The neural P-representation beats appearance-only grouping under visual shift.
- The neural P-representation beats the prior-average action model.
- The predicted-test representation recovers hidden action regimes with high purity.
- The neural P-representation approaches the engineered fingerprint reference score.
- Hidden labels are not used as learner features.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.neural_claims
```

## Neural Sample-Efficiency Evidence

`simulation/pjepa_sim/representation/neural.py` also tests sparse sampled intervention evidence by varying the number of intervention repeats per context while holding the context stream fixed.

Current executable checks:

- The neural P-representation beats the prior baseline at every tested repeat count.
- The neural P-representation beats appearance-only grouping at every tested repeat count.
- The learned predicted-test vectors keep high regime purity across repeat counts.
- The neural P-representation approaches the engineered fingerprint reference across repeat counts.
- Prediction error decreases as intervention repeats increase.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.neural_sample_efficiency_claims
```

## Neural Active-Probing Evidence

`simulation/pjepa_sim/representation/neural_active.py` tests whether a learned predicted-test model can use safe probes when initial structured sensors alias hidden regimes. The learner receives low-dimensional sensor features, probe-evidence features, and test identities; hidden labels remain outside learner inputs.

Current executable checks:

- Learned active probing beats acting immediately from the ambiguous initial observation.
- Learned active probing reduces unsafe failure relative to no probing.
- Learned value-aware probing beats a learned entropy-probing baseline.
- The active policy actually uses probes.
- Learned active probing recovers much of the hidden-regime oracle value.
- Hidden labels are not used as learner features.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.neural_active_probe_claims
```

## Neural Active-Probing Boundary Evidence

`simulation/pjepa_sim/representation/neural_active.py` also tests the boundary conditions for that claim. The sweep compares aliased versus distinct initial sensors, informative versus weak probe likelihoods, and cheap versus costly probes.

Current executable checks:

- Active probing has a large score margin when sensors alias regimes and probes are informative.
- The active-probing margin is smaller when initial sensors already identify the regime.
- Weak probes reduce the active-probing margin.
- Costly probes cause the value-aware policy to use less than the full probe budget.
- Value-aware probing slightly beats entropy probing in the aliased informative setting.
- Distinct sensors require fewer representation-repair probes than aliased sensors.
- Hidden labels are not used as learner features.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.neural_active_boundary_claims
```

## Neural Active-Probing Seed-Sweep Evidence

`simulation/pjepa_sim/representation/neural_active.py` repeats the aliased-sensor, informative-probe learned active-probing benchmark over multiple deterministic seeds. This is not a statistical confidence interval, but it checks that the headline safety and no-probe margins are not a one-seed artifact.

Current executable checks:

- Active probing beats no probing on average.
- Active probing beats no probing for every tested seed.
- Active probing reduces unsafe failure on average.
- Active probing reduces unsafe failure for every tested seed.
- Value-aware probing is entropy-competitive on average.
- No tested seed shows a large regression against entropy probing.
- Learned active probing remains close to the hidden-regime oracle on average.
- Hidden labels are not used as learner features.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.neural_active_seed_sweep_claims
```

## Pixel Continuous-Control Evidence

`simulation/pjepa_sim/perception/continuous.py` is the first local test that removes structured sensor vectors from the learned active-probing path. It renders small pixel observations that alias pairs of hidden continuous-control regimes, then evaluates continuous 2D reach-controller rollouts. The learner receives pixels, probe-evidence features, and test identities; hidden labels remain outside learner inputs.

Current executable checks:

- Pixel active probing beats no probing on risk-adjusted score.
- Pixel active probing reduces unsafe continuous-control failures.
- Pixel active probing remains competitive with entropy probing.
- Pixel active probing actually uses probes.
- Pixel active probing recovers some oracle value while leaving substantial headroom.
- Hidden labels are not used as learner features.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.pixel_continuous_claims
```

## Video Representation Surrogate Evidence

`simulation/pjepa_sim/perception/video_representation.py` tests the first local version of the JEPA comparison. A passive video predictor learns to predict future rendered frames from context frames, while the P-representation learner clusters sampled intervention consequences. The benchmark intentionally shifts visual styles between train and test, so passive visual prediction can remain accurate while action-regime identity becomes unstable.

Current executable checks:

- The action-conditioned representation beats the passive video representation on downstream risk-adjusted score.
- The action-conditioned representation beats the prior-average action model.
- The passive video predictor has low future-frame prediction error on its own objective.
- The passive video representation has low action-regime purity under visual shift.
- The action-conditioned representation has high action-regime purity.
- The action-conditioned representation approaches the oracle regime score.
- Hidden labels are not used as learner features.
- The report explicitly records that no actual V-JEPA or video foundation model was run.
- Corrupting or permuting intervention evidence destroys most of the action-representation advantage.
- Increasing sampled intervention repeats reduces action-feature error.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.video_representation_claims
```

## Load-Bearing Real-Video Evidence

`simulation/pjepa_sim/real_video/kth_samples.py` is the first non-generated video check. It downloads the six official sample AVI files from the KTH action database (Schuldt, Laptev & Caputo, 2004), decodes them with `ffmpeg`, segments them into temporal windows, and compares static appearance, passive next-frame, and temporal-motion descriptors.

Current result on this sample split:

- Static appearance accuracy: `0.896`.
- Passive next-frame descriptor accuracy: `0.805`.
- Temporal motion descriptor accuracy: `0.623`.

This is a negative/diagnostic result for the current P-JEPA paper, not a win. The sample split is dominated by appearance/background identity. It is nevertheless load-bearing: it is part of `verify_all` and prevents the paper from treating synthetic video as sufficient. It shows that actual video benchmarking can contradict the local surrogate and that a serious claim requires a proper action-video or robot-video dataset with train/test separation by subject, scene, and ideally intervention/action metadata.

The code now includes a manifest-based full-video protocol in `simulation/pjepa_sim/real_video/manifest_benchmark.py` and a KTH manifest builder in `simulation/pjepa_sim/real_video/manifest_builders.py`. This is not a new performance result. It is an audit guard for the next validity test: a full real-video benchmark must use actual files, must not place the same file in train and test, must cover the same classes across splits, must supply subject/scene/source groups for disjointness, and must include action/intervention metadata before it can support a P-JEPA action-grounding claim. The protocol verifier also checks that the six-file KTH sample set is rejected as a full train/test benchmark.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download
uv run python -m pjepa_sim.verification.kth_sample_video_claims
uv run python -m pjepa_sim.verification.manifest_video_protocol_claims
```

## Robot-Policy Protocol Evidence

`simulation/pjepa_sim/robot/manifest_protocol.py` is not a robot-learning result. It is a claim guard for the next stage. A future robot-policy benchmark must supply episode observations, action trajectories, task labels, split labels, group ids for leakage control, success metrics, and unsafe-failure metrics. Optional checks require language metadata and robot/embodiment metadata.

Current executable checks:

- A complete robot-policy manifest is accepted.
- A manifest without action trajectories is rejected.
- A manifest without task-success metrics is rejected.
- A manifest without unsafe-failure metrics is rejected by default.
- Non-safety runs can explicitly disable the unsafe-metric requirement.
- Group leakage between train and test is rejected.
- Task-incomplete train/test splits are rejected.
- Missing observation files are rejected.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.robot_manifest_protocol_claims
```

## Evidence-Level Guard

`simulation/pjepa_sim/verification/evidence.py` classifies every local verifier by evidence level. `simulation/pjepa_sim/verification/evidence_claims.py` checks that every local verifier has a classification, that protocol checks are not counted as performance evidence, that the KTH sample remains diagnostic negative evidence, that the video surrogate is not treated as V-JEPA evidence, and that the robot manifest protocol is not treated as a robot-policy result.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.evidence_claims
```

## Formal Contract-Interface Evidence

`simulation/pjepa_sim/formal/contracts.py` tests whether P-JEPA outputs can be exported as finite contracts for safety and verification systems. The current local checker bounds expected unsafe failure, worst hidden-regime branch unsafe failure, residual obstruction, probe budget, and risk-adjusted score across the configured suites. This is a Kona/Aleph adapter protocol, not a Kona/Aleph run.

Current executable checks:

- P-JEPA satisfies all finite contracts across the configured local suites.
- P-JEPA satisfies more finite contracts than the prior predictive baseline.
- P-JEPA satisfies more finite contracts than entropy probing under the chosen safety-efficiency contract.
- The checker returns machine-readable counterexamples for the prior baseline.
- Every suite-agent pair exports one machine-readable contract artifact.
- The report explicitly records that no external Kona or Aleph backend was executed.

Verifier:

```bash
cd simulation
uv run python -m pjepa_sim.verification.formal_contract_claims
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

The current implementation does not demonstrate learning a robot controller, tactile representation learning, language grounding, online regime discovery from uncontrolled logs, multi-task robot transfer, neural sheaf learning, a uniquely sheaf-theoretic advantage isolated from intervention, viability, and value-of-information effects, or scalability to internet video or foundation-model pretraining. The neural benchmarks learn from low-dimensional structured sensor features, probe-evidence features, and test identities, not tactile streams or end-to-end robot trajectories; their sample-efficiency, active-probing, boundary-condition, and seed-sweep results are toy results, not general data-efficiency or robotics claims. The pixel continuous-control benchmark is the first local move toward learned perception and harder control, but it still uses tiny rendered images, a small MLP, finite controller templates, and simulated 2D dynamics. It is not MuJoCo-scale robot learning. The video-representation benchmark is a local surrogate for passive JEPA-style prediction; it is not a result against V-JEPA, V-JEPA 2, or any video foundation model. The KTH sample check is real video but diagnostic negative evidence, not a P-JEPA video win. The manifest protocols and evidence-level guard are infrastructure, not performance evidence. The formal contract-interface benchmark does not run Kona, Aleph, Lean, or any external theorem prover; it is a finite local checker and export protocol for future proof/constraint backends. The active-probing boundary benchmark is useful precisely because it shows when the claim weakens: if sensors already identify the regime, probing has no marginal value; if probes are weak, the learned active-probing gain nearly disappears. The seed sweep supports the no-probe and unsafe-failure claims across tested deterministic seeds, but it also shows a nuance: value-aware probing is only slightly better than entropy probing on average and can lose to entropy on an individual seed. The synthetic scaling benchmark is a controlled regime-count sweep over engineered action-consequence fingerprints, not evidence of high-dimensional neural scaling. The gluing ablation learns linear restriction maps over engineered local section vectors; it is not an end-to-end neural sheaf. The representation, online cover-construction, and composition benchmarks use engineered action/probe fingerprints and skill tables rather than learned sensory encoders or learned options.

These limits are not defects to hide. They define the correct scientific status: formal proposal plus controlled executable demonstrations.

## Claim Hygiene

- Do not cite internal PIATRA papers as evidence.
- Do not turn a toy benchmark into a robotics claim.
- Do not report generated numbers without a command that regenerates them.
- Do not compare against weak baselines only; keep entropy and random probing in the comparison.
- Keep the distinction between pure obstruction reduction and the full viability-aware P-JEPA stack explicit.

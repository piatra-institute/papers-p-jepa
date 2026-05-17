# Implementation Guide

This document explains how the simulation code is organized and how to extend it without weakening the scientific claims.

## Module Map

### Exact Prototype

- `simulation/pjepa_sim/core/dishworld.py`: defines regimes, actions, probe likelihoods, Bayesian update, obstruction, and expected action outcomes.
- `simulation/pjepa_sim/core/agents.py`: defines the original exact policies used by `pjepa_sim.cli.run_all`. This file is kept simple because it backs the earliest paper figures and the original obstruction-only checks.
- `simulation/pjepa_sim/cli/run_all.py`: runs the original exact simulation and writes `output/results.json` plus figures.
- `simulation/pjepa_sim/core/figures.py`: generates figures for the original exact simulation.

### Benchmark Suite

- `simulation/pjepa_sim/benchmark/suites.py`: the main exact evaluator. It supports configurable suites, exact evidence-tree evaluation, risk-adjusted scoring, posterior-entropy probing, obstruction probing, distinct world/belief local-section models, and the full P-JEPA stack.
- `simulation/pjepa_sim/cli/benchmark.py`: CLI implementation for running benchmark suites and writing JSON, Markdown, and summary figures.
- `simulation/pjepa_sim/benchmark/configs/*.json`: benchmark suite definitions. Add new suites here when the mechanism should be tested under a new condition.

### Representation Benchmark

- `simulation/pjepa_sim/representation/clustering.py`: shared deterministic standardisation and k-means helpers used by the representation benchmarks.
- `simulation/pjepa_sim/representation/learning.py`: generates shifted visual/action contexts, clusters unlabeled action/probe fingerprints, fits local action sections, and evaluates downstream action choice.
- `simulation/pjepa_sim/representation/neural.py`: trains a small deterministic NumPy MLP from sampled intervention records, uses the predicted-test vector as a learned P-representation, and runs the neural sample-efficiency sweep.
- `simulation/pjepa_sim/representation/neural_active.py`: trains a small NumPy MLP with probe-evidence features and evaluates learned active probing under ambiguous structured sensors, including a boundary-condition sweep over sensor aliasing, probe informativeness, probe cost, and a seed-robustness sweep.
- `simulation/pjepa_sim/representation/online.py`: builds an action-consequence cover incrementally from an unlabeled stream and evaluates the discovered local sections under shifted visual labels.
- `simulation/pjepa_sim/representation/scaling.py`: generates synthetic hidden-regime sweeps over 4, 8, 16, and 32 action regimes and evaluates whether action-consequence covers remain useful.
- `simulation/pjepa_sim/representation/gluing.py`: generates local action sections in incompatible action-coordinate frames, learns restriction maps from overlap records, and compares glued aggregation against identity/no-glue aggregation.
- `simulation/pjepa_sim/representation/composition.py`: generates two-step skill-chain contexts, learns action-consequence clusters, fits local chain sections, and evaluates precondition/postcondition composition.
- `simulation/pjepa_sim/cli/representation_benchmark.py`: CLI implementation for writing `output/representation_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/neural_benchmark.py`: CLI implementation for writing `output/neural_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/neural_sample_efficiency_benchmark.py`: CLI implementation for writing `output/neural_sample_efficiency_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/neural_active_probe_benchmark.py`: CLI implementation for writing `output/neural_active_probe_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/neural_active_boundary_benchmark.py`: CLI implementation for writing `output/neural_active_boundary_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/neural_active_seed_sweep_benchmark.py`: CLI implementation for writing `output/neural_active_seed_sweep_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/online_cover_benchmark.py`: CLI implementation for writing `output/online_cover_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/scaling_benchmark.py`: CLI implementation for writing `output/scaling_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/gluing_ablation_benchmark.py`: CLI implementation for writing `output/gluing_ablation_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/skill_composition_benchmark.py`: CLI implementation for writing `output/skill_composition_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/verification/representation_claims.py`: checks that action-consequence grouping beats appearance grouping and prior averaging under visual shift.
- `simulation/pjepa_sim/verification/neural_claims.py`: checks that the learned intervention encoder beats appearance and prior baselines and approaches the engineered fingerprint reference.
- `simulation/pjepa_sim/verification/neural_sample_efficiency_claims.py`: checks that the learned intervention encoder remains useful across sparse intervention-repeat budgets.
- `simulation/pjepa_sim/verification/neural_active_probe_claims.py`: checks that learned active probing improves score and safety when initial structured sensors alias hidden regimes.
- `simulation/pjepa_sim/verification/neural_active_boundary_claims.py`: checks that learned active probing has the expected boundary conditions under sensor aliasing, weak probes, and costly probes.
- `simulation/pjepa_sim/verification/neural_active_seed_sweep_claims.py`: checks that the learned active-probing no-probe margin and unsafe reduction are robust across deterministic seeds.
- `simulation/pjepa_sim/verification/online_claims.py`: checks that online cover construction discovers the intended action regimes and preserves the representation benchmark advantage.
- `simulation/pjepa_sim/verification/scaling_claims.py`: checks that action-consequence grouping remains above appearance and prior baselines throughout the synthetic regime-count sweep.
- `simulation/pjepa_sim/verification/gluing_claims.py`: checks that learned restriction maps reduce overlap residual and improve action choice relative to identity/no-glue aggregation.
- `simulation/pjepa_sim/verification/composition_claims.py`: checks that action-consequence grouping composes the intended skill chains under visual shift.

### External Adapter

- `simulation/pjepa_sim/cli/external_benchmark.py`: CLI implementation for optional Meta-World runs.
- `simulation/pjepa_sim/external/metaworld_hidden_regime.py`: hidden-regime Meta-World wrapper, scripted policies, probe strategies, and strategy aggregation.
- `simulation/pjepa_sim/external/learned_metaworld.py`: learned probe likelihoods, learned local sections, unsupervised clustering, prior-stream collection, raw-record collection, and learned-model benchmark runners.

### Real Video

- `simulation/pjepa_sim/real_video/kth_samples.py`: load-bearing KTH sample-video smoke test using real downloaded AVI files and `ffmpeg` decoding.
- `simulation/pjepa_sim/real_video/manifest_benchmark.py`: manifest-based real-video benchmark runner for full datasets with train/test split, group-disjointness, same-file leakage, and action-metadata validation.
- `simulation/pjepa_sim/real_video/manifest_builders.py`: dataset-specific manifest builders. The current builder parses KTH action-database filenames into subject-grouped train/test records.
- `simulation/pjepa_sim/cli/kth_sample_video_benchmark.py`: CLI implementation for downloading the six official KTH sample videos and writing `output/kth_sample_video_benchmark.json`.
- `simulation/pjepa_sim/cli/manifest_video_benchmark.py`: CLI implementation for running a manifest-defined real-video benchmark with leakage-aware validation.
- `simulation/pjepa_sim/cli/prepare_video_manifest.py`: CLI implementation for preparing dataset manifests, currently including KTH filename parsing.
- `simulation/pjepa_sim/verification/kth_sample_video_claims.py`: load-bearing verifier for the KTH sample-video result. It checks that the benchmark uses real video files and records the current negative result that static/passive descriptors beat temporal motion on this appearance-dominated sample split.
- `simulation/pjepa_sim/verification/manifest_video_protocol_claims.py`: verifier for the full-video protocol. It checks that the manifest runner rejects class-incomplete splits, same-video train/test leakage, missing group metadata, and missing action metadata when action-grounding claims are requested. It also checks that the KTH manifest builder parses sample filenames while refusing to treat the six-file sample as a full split.

### Verification

- `simulation/pjepa_sim/verification/reporting.py`: shared helpers for margin-based verifier reports.
- `simulation/pjepa_sim/verification/audit.py`: local verifier registry and generated claims-summary writer.
- `simulation/pjepa_sim/verification/exact_claims.py`: original exact obstruction-only checks.
- `simulation/pjepa_sim/verification/benchmark_claims.py`: suite-level checks for the full P-JEPA stack.
- `simulation/pjepa_sim/verification/representation_claims.py`: representation-learning checks for action-consequence clustering under visual shift.
- `simulation/pjepa_sim/verification/neural_claims.py`: neural intervention encoder checks for learned predicted-test representations.
- `simulation/pjepa_sim/verification/neural_sample_efficiency_claims.py`: neural sample-efficiency checks for sparse sampled intervention evidence.
- `simulation/pjepa_sim/verification/neural_active_probe_claims.py`: neural active-probing checks for learned value of information under ambiguous observations.
- `simulation/pjepa_sim/verification/neural_active_boundary_claims.py`: neural active-probing boundary checks for when the active-probing claim should strengthen or weaken.
- `simulation/pjepa_sim/verification/neural_active_seed_sweep_claims.py`: neural active-probing seed-sweep checks for robustness of the aliased-sensor result.
- `simulation/pjepa_sim/verification/pixel_continuous_claims.py`: pixel-observation continuous-control checks for the first local learned-perception stress test.
- `simulation/pjepa_sim/verification/video_representation_claims.py`: video-surrogate checks for passive next-frame prediction versus action-conditioned predicted-test representations.
- `simulation/pjepa_sim/verification/kth_sample_video_claims.py`: load-bearing real-video smoke-test checks over downloaded KTH sample AVI files.
- `simulation/pjepa_sim/verification/manifest_video_protocol_claims.py`: leakage and metadata protocol checks for future full real-video benchmarks.
- `simulation/pjepa_sim/verification/formal_contract_claims.py`: verification-interface checks for finite safety contracts and counterexample reporting.
- `simulation/pjepa_sim/verification/online_claims.py`: online cover-construction checks for incremental action-consequence regime discovery.
- `simulation/pjepa_sim/verification/scaling_claims.py`: synthetic scaling checks for action-grounded representation as hidden regime count increases.
- `simulation/pjepa_sim/verification/gluing_claims.py`: restriction-map ablation checks for local-to-global consistency.
- `simulation/pjepa_sim/verification/composition_claims.py`: skill-composition checks for precondition/postcondition chains.
- `simulation/pjepa_sim/verification/external_claims.py`: hand-specified Meta-World adapter checks.
- `simulation/pjepa_sim/verification/learned_external_claims.py`: supervised learned-model checks.
- `simulation/pjepa_sim/verification/unsupervised_external_claims.py`: balanced unsupervised checks.
- `simulation/pjepa_sim/verification/stream_external_claims.py`: prior-stream unsupervised checks.
- `simulation/pjepa_sim/verification/raw_record_external_claims.py`: raw-record learner checks.
- `simulation/pjepa_sim/cli/verify_all.py`: runs all local verifiers and writes `output/claims_summary.json` plus `output/CLAIMS_SUMMARY.md`.

### Perception and Continuous Control

- `simulation/pjepa_sim/perception/continuous.py`: renders small image observations, simulates hidden continuous 2D action dynamics, trains a pixel/evidence/test MLP, and evaluates no-probe, entropy-probe, active-probe, and oracle policies.
- `simulation/pjepa_sim/cli/pixel_continuous_benchmark.py`: CLI implementation for writing `output/pixel_continuous_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/perception/video_representation.py`: renders short passive video contexts, trains a passive next-frame predictor, and compares its downstream representation against action-conditioned predicted-test features under visual shift.
- `simulation/pjepa_sim/cli/video_representation_benchmark.py`: CLI implementation for writing `output/video_representation_benchmark.json` and the Markdown table.

### Formal Verification Interface

- `simulation/pjepa_sim/formal/contracts.py`: exports finite policy contracts for safety, branch safety, risk-adjusted score, residual obstruction, and probe budget; also provides the local deterministic checker and counterexample records.
- `simulation/pjepa_sim/cli/formal_contract_benchmark.py`: CLI implementation for writing `output/formal_contract_benchmark.json` and the Markdown table.

This interface is the right integration point for Kona/Aleph-style systems. The benchmark artifact is machine-readable and finite; an external verifier can prove the same requirements or return counterexamples. The repository currently uses only the local checker and explicitly records that no Kona or Aleph backend was executed.

## Adding a New Exact Suite

1. Add a JSON file under `simulation/pjepa_sim/benchmark/configs/`.
2. Include a new suite name in `SUITE_ORDER` if order matters.
3. Use `belief_action_model` or `belief_probe_likelihood` when the policy should act from a learned or miscalibrated model while the true `action_model` and `probe_likelihood` remain the evaluation world.
4. Run:

```bash
uv run python -m pjepa_sim.cli.benchmark --suite all --agents all
uv run python -m pjepa_sim.verification.benchmark_claims
```

5. If claims change, update `paper/PAPER.md`, this documentation, and rebuild `paper/PAPER.pdf`.

## Adding a New Agent

1. Add the evaluator to `pjepa_sim/benchmark/suites.py`.
2. Add the agent name to `pjepa_sim/cli/benchmark.py` if it should be part of `--agents all`.
3. Add verifier checks only if the paper will claim that the agent establishes a comparison.
4. Update the paper tables only after regenerating output.

## Adding a New External Strategy

1. Add a probe strategy in `pjepa_sim/external/metaworld_hidden_regime.py`.
2. Add a `StrategySpec` if it should be benchmarked.
3. Regenerate the relevant external outputs.
4. Update the relevant verifier to assert the intended comparison.

## Generated Artifacts

`simulation/output/` is generated and ignored by git. The directory contains result JSON files, Markdown tables, verifier reports, and figures.

Do not treat these files as source code. Regenerate them before changing paper numbers.

## Numerical Discipline

The project is only as strong as the link between code and claims. Follow this rule:

```text
no numeric paper claim without a JSON source and verifier or an explicit caveat
```

When a command changes a number, update all dependent text in one pass.

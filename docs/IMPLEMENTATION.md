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
- `simulation/pjepa_sim/representation/online.py`: builds an action-consequence cover incrementally from an unlabeled stream and evaluates the discovered local sections under shifted visual labels.
- `simulation/pjepa_sim/representation/scaling.py`: generates synthetic hidden-regime sweeps over 4, 8, 16, and 32 action regimes and evaluates whether action-consequence covers remain useful.
- `simulation/pjepa_sim/representation/gluing.py`: generates local action sections in incompatible action-coordinate frames, learns restriction maps from overlap records, and compares glued aggregation against identity/no-glue aggregation.
- `simulation/pjepa_sim/representation/composition.py`: generates two-step skill-chain contexts, learns action-consequence clusters, fits local chain sections, and evaluates precondition/postcondition composition.
- `simulation/pjepa_sim/cli/representation_benchmark.py`: CLI implementation for writing `output/representation_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/online_cover_benchmark.py`: CLI implementation for writing `output/online_cover_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/scaling_benchmark.py`: CLI implementation for writing `output/scaling_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/gluing_ablation_benchmark.py`: CLI implementation for writing `output/gluing_ablation_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/cli/skill_composition_benchmark.py`: CLI implementation for writing `output/skill_composition_benchmark.json` and the Markdown table.
- `simulation/pjepa_sim/verification/representation_claims.py`: checks that action-consequence grouping beats appearance grouping and prior averaging under visual shift.
- `simulation/pjepa_sim/verification/online_claims.py`: checks that online cover construction discovers the intended action regimes and preserves the representation benchmark advantage.
- `simulation/pjepa_sim/verification/scaling_claims.py`: checks that action-consequence grouping remains above appearance and prior baselines throughout the synthetic regime-count sweep.
- `simulation/pjepa_sim/verification/gluing_claims.py`: checks that learned restriction maps reduce overlap residual and improve action choice relative to identity/no-glue aggregation.
- `simulation/pjepa_sim/verification/composition_claims.py`: checks that action-consequence grouping composes the intended skill chains under visual shift.

### External Adapter

- `simulation/pjepa_sim/cli/external_benchmark.py`: CLI implementation for optional Meta-World runs.
- `simulation/pjepa_sim/external/metaworld_hidden_regime.py`: hidden-regime Meta-World wrapper, scripted policies, probe strategies, and strategy aggregation.
- `simulation/pjepa_sim/external/learned_metaworld.py`: learned probe likelihoods, learned local sections, unsupervised clustering, prior-stream collection, raw-record collection, and learned-model benchmark runners.

### Verification

- `simulation/pjepa_sim/verification/reporting.py`: shared helpers for margin-based verifier reports.
- `simulation/pjepa_sim/verification/exact_claims.py`: original exact obstruction-only checks.
- `simulation/pjepa_sim/verification/benchmark_claims.py`: suite-level checks for the full P-JEPA stack.
- `simulation/pjepa_sim/verification/representation_claims.py`: representation-learning checks for action-consequence clustering under visual shift.
- `simulation/pjepa_sim/verification/online_claims.py`: online cover-construction checks for incremental action-consequence regime discovery.
- `simulation/pjepa_sim/verification/scaling_claims.py`: synthetic scaling checks for action-grounded representation as hidden regime count increases.
- `simulation/pjepa_sim/verification/gluing_claims.py`: restriction-map ablation checks for local-to-global consistency.
- `simulation/pjepa_sim/verification/composition_claims.py`: skill-composition checks for precondition/postcondition chains.
- `simulation/pjepa_sim/verification/external_claims.py`: hand-specified Meta-World adapter checks.
- `simulation/pjepa_sim/verification/learned_external_claims.py`: supervised learned-model checks.
- `simulation/pjepa_sim/verification/unsupervised_external_claims.py`: balanced unsupervised checks.
- `simulation/pjepa_sim/verification/stream_external_claims.py`: prior-stream unsupervised checks.
- `simulation/pjepa_sim/verification/raw_record_external_claims.py`: raw-record learner checks.

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

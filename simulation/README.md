# P-JEPA Simulation

This directory contains the executable evidence for the P-JEPA paper. The code tests hidden-regime action worlds where visually similar situations differ in action consequences.

For conceptual background, see:

- `../docs/ARCHITECTURE.md`
- `../docs/IMPLEMENTATION.md`
- `../docs/SCIENTIFIC_CLAIMS.md`
- `../docs/REPRODUCIBILITY.md`

## Core Commands

Run the original exact simulation:

```bash
uv run python -m pjepa_sim.cli.run_all
uv run python -m pjepa_sim.verification.exact_claims
```

Run the suite-level benchmark:

```bash
uv run python -m pjepa_sim.cli.benchmark --suite all --agents all
uv run python -m pjepa_sim.verification.benchmark_claims
```

Run the action-grounded representation benchmark:

```bash
uv run python -m pjepa_sim.cli.representation_benchmark
uv run python -m pjepa_sim.verification.representation_claims
```

Run the online cover-construction benchmark:

```bash
uv run python -m pjepa_sim.cli.online_cover_benchmark
uv run python -m pjepa_sim.verification.online_claims
```

Run the synthetic representation-scaling benchmark:

```bash
uv run python -m pjepa_sim.cli.scaling_benchmark
uv run python -m pjepa_sim.verification.scaling_claims
```

Run the restriction-map gluing ablation:

```bash
uv run python -m pjepa_sim.cli.gluing_ablation_benchmark
uv run python -m pjepa_sim.verification.gluing_claims
```

Run the skill-composition benchmark:

```bash
uv run python -m pjepa_sim.cli.skill_composition_benchmark
uv run python -m pjepa_sim.verification.composition_claims
```

Check the optional Meta-World adapter:

```bash
uv run python -m pjepa_sim.cli.external_benchmark --check
```

Run the strongest current external result, the raw-record learner:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-raw-record-benchmark --episodes 100 --stream-contexts 160 --unsupervised-probe-trials 16 --unsupervised-action-trials 64
uv run python -m pjepa_sim.verification.raw_record_external_claims
```

## Other External Runs

Hand-specified adapter:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-benchmark --episodes 100
uv run python -m pjepa_sim.verification.external_claims
```

Supervised learned model:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-learned-benchmark --episodes 100 --train-probe-samples 64 --train-action-samples 256
uv run python -m pjepa_sim.verification.learned_external_claims
```

Balanced unsupervised model:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-unsupervised-benchmark --episodes 100 --unsupervised-contexts 32 --unsupervised-probe-trials 16 --unsupervised-action-trials 64
uv run python -m pjepa_sim.verification.unsupervised_external_claims
```

Prior-stream unsupervised model:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-stream-benchmark --episodes 100 --stream-contexts 160 --unsupervised-probe-trials 16 --unsupervised-action-trials 64
uv run python -m pjepa_sim.verification.stream_external_claims
```

## Layout

```text
simulation/
├── pjepa_sim/
│   ├── benchmark/                    suite evaluator and P-JEPA stack
│   │   └── configs/                  benchmark suite definitions
│   ├── cli/                          CLI implementations
│   ├── core/                         exact hidden-regime model and original agents
│   ├── external/                     Meta-World adapter and learned estimators
│   ├── representation/               action-grounded representation benchmark
│   ├── verification/                 executable claim checks
│   └── paths.py                      repository-local paths
├── pyproject.toml                    uv project metadata
├── uv.lock                           locked Python environment
└── output/                           generated, gitignored artifacts
```

## What Is Being Tested

The exact toy world uses one visible object class and four hidden regimes: `dry`, `soapy`, `cracked`, and `heavy`. The hidden regime changes the success and unsafe-failure probabilities of task actions.

The benchmark compares no-probe predictive baselines, posterior-entropy probing, pure obstruction reduction, viability-aware P-JEPA probing, and oracle hidden-regime access. One suite also separates the policy's noisy belief sections from the true world sections to test a minimal learned-model mismatch. The representation benchmark separately tests whether unlabeled action/probe fingerprints support downstream action choice better than unstable visual grouping. The online cover-construction benchmark tests whether the same local regimes can be built incrementally from an unlabeled stream. The scaling benchmark varies the number of synthetic hidden action regimes from 4 to 32 while keeping visual labels low-cardinality and shifted. The gluing ablation tests whether learned restriction maps help when local action sections use incompatible action-coordinate frames. The skill-composition benchmark tests whether those action-grounded representations support two-step precondition/postcondition chains.

The Meta-World adapter repeats the same comparison around a scripted continuous-control `reach-v3` controller. It is a probe-selection and hidden-regime inference benchmark, not a learned robotics policy benchmark.

## Output Policy

`output/` is generated and ignored by git. Regenerate it with the commands above before changing paper numbers.

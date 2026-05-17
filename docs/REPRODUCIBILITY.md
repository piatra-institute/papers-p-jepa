# Reproducibility

This document lists the commands needed to regenerate the project results.

## Environment

The simulation uses `uv`.

```bash
cd simulation
uv run python -V
```

The exact benchmark depends on Python 3.10 or newer, `numpy`, and `matplotlib`.

The optional external adapter additionally requires `gymnasium`, `metaworld`, and MuJoCo runtime dependencies.

External commands use `uv run --with gymnasium --with metaworld python -m pjepa_sim...` so the optional dependencies do not need to be permanent project dependencies.

## Exact Simulation

```bash
cd simulation
uv run python -m pjepa_sim.cli.run_all
uv run python -m pjepa_sim.verification.exact_claims
```

Generated files:

- `output/results.json`
- `output/verification.json`
- `output/figures/agent_outcomes.png`
- `output/figures/obstruction_reduction.png`
- `output/figures/hidden_regime_transfer.png`

## Benchmark Suites

```bash
cd simulation
uv run python -m pjepa_sim.cli.benchmark --suite all --agents all
uv run python -m pjepa_sim.verification.benchmark_claims
```

Generated files:

- `output/benchmark_results.json`
- `output/benchmark_table.md`
- `output/benchmark_verification.json`
- `output/figures/benchmark_summary.png`

## Representation Benchmark

```bash
cd simulation
uv run python -m pjepa_sim.cli.representation_benchmark
uv run python -m pjepa_sim.verification.representation_claims
```

Generated files:

- `output/representation_benchmark.json`
- `output/representation_benchmark.md`
- `output/representation_verification.json`

## Online Cover-Construction Benchmark

```bash
cd simulation
uv run python -m pjepa_sim.cli.online_cover_benchmark
uv run python -m pjepa_sim.verification.online_claims
```

Generated files:

- `output/online_cover_benchmark.json`
- `output/online_cover_benchmark.md`
- `output/online_cover_verification.json`

## Synthetic Scaling Benchmark

```bash
cd simulation
uv run python -m pjepa_sim.cli.scaling_benchmark
uv run python -m pjepa_sim.verification.scaling_claims
```

Generated files:

- `output/scaling_benchmark.json`
- `output/scaling_benchmark.md`
- `output/scaling_verification.json`

## Restriction-Map Gluing Ablation

```bash
cd simulation
uv run python -m pjepa_sim.cli.gluing_ablation_benchmark
uv run python -m pjepa_sim.verification.gluing_claims
```

Generated files:

- `output/gluing_ablation_benchmark.json`
- `output/gluing_ablation_benchmark.md`
- `output/gluing_ablation_verification.json`

## Skill-Composition Benchmark

```bash
cd simulation
uv run python -m pjepa_sim.cli.skill_composition_benchmark
uv run python -m pjepa_sim.verification.composition_claims
```

Generated files:

- `output/skill_composition_benchmark.json`
- `output/skill_composition_benchmark.md`
- `output/skill_composition_verification.json`

## External Adapter

Check availability:

```bash
cd simulation
uv run python -m pjepa_sim.cli.external_benchmark --check
```

Run the hand-specified adapter benchmark:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-benchmark --episodes 100
uv run python -m pjepa_sim.verification.external_claims
```

Run supervised learned fitting:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-learned-benchmark --episodes 100 --train-probe-samples 64 --train-action-samples 256
uv run python -m pjepa_sim.verification.learned_external_claims
```

Run balanced unsupervised fitting:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-unsupervised-benchmark --episodes 100 --unsupervised-contexts 32 --unsupervised-probe-trials 16 --unsupervised-action-trials 64
uv run python -m pjepa_sim.verification.unsupervised_external_claims
```

Run prior-stream unsupervised fitting:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-stream-benchmark --episodes 100 --stream-contexts 160 --unsupervised-probe-trials 16 --unsupervised-action-trials 64
uv run python -m pjepa_sim.verification.stream_external_claims
```

Run raw-record fitting:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-raw-record-benchmark --episodes 100 --stream-contexts 160 --unsupervised-probe-trials 16 --unsupervised-action-trials 64
uv run python -m pjepa_sim.verification.raw_record_external_claims
```

## Paper Build

```bash
./paper/scripts/build-paper.sh
```

The build writes `paper/PAPER.pdf`. The PDF is generated and ignored by git.

## Generated Output Policy

`simulation/output/` and `paper/PAPER.pdf` are ignored by git. They are reproducible output, not source.

If reviewers need a result snapshot, regenerate the output directory with the commands above rather than relying on stale checked-in artifacts.

## Expected Warning

Some Meta-World/Gymnasium runs print observation-space warnings from the wrapped environment. They have been observed in successful runs and are not currently treated as verifier failures. The verifier scripts are the authority for pass or fail.

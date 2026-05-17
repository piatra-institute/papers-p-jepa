# Reproducibility

This document lists the commands needed to regenerate the project results.

## Environment

The simulation uses `uv`.

```bash
cd simulation
uv run python -V
```

The exact benchmark depends on Python 3.10 or newer, `numpy`, and `matplotlib`.

The local audit also requires the six official KTH sample AVI files. Download them once with `uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download`; the files are written under `simulation/data/kth_samples/`, which is gitignored.

The optional external adapter additionally requires `gymnasium`, `metaworld`, and MuJoCo runtime dependencies.

External commands use `uv run --with gymnasium --with metaworld python -m pjepa_sim...` so the optional dependencies do not need to be permanent project dependencies.

## Local Verification Audit

```bash
cd simulation
uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download
uv run python -m pjepa_sim.cli.verify_all
```

Generated files:

- `output/claims_summary.json`
- `output/CLAIMS_SUMMARY.md`
- `output/evidence_matrix.json`
- `output/EVIDENCE_MATRIX.md`

This command runs all local verifiers listed in `pjepa_sim.verification.audit.LOCAL_VERIFIERS`, including the KTH sample real-video smoke test. It deliberately excludes optional Meta-World checks because those require external simulator dependencies.

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

## Neural Intervention Encoder Benchmark

```bash
cd simulation
uv run python -m pjepa_sim.cli.neural_benchmark
uv run python -m pjepa_sim.verification.neural_claims
```

Generated files:

- `output/neural_benchmark.json`
- `output/neural_benchmark.md`
- `output/neural_verification.json`

## Neural Sample-Efficiency Benchmark

```bash
cd simulation
uv run python -m pjepa_sim.cli.neural_sample_efficiency_benchmark
uv run python -m pjepa_sim.verification.neural_sample_efficiency_claims
```

Generated files:

- `output/neural_sample_efficiency_benchmark.json`
- `output/neural_sample_efficiency_benchmark.md`
- `output/neural_sample_efficiency_verification.json`

## Neural Active-Probing Benchmark

```bash
cd simulation
uv run python -m pjepa_sim.cli.neural_active_probe_benchmark
uv run python -m pjepa_sim.verification.neural_active_probe_claims
```

Generated files:

- `output/neural_active_probe_benchmark.json`
- `output/neural_active_probe_benchmark.md`
- `output/neural_active_probe_verification.json`

## Neural Active-Probing Boundary Benchmark

```bash
cd simulation
uv run python -m pjepa_sim.cli.neural_active_boundary_benchmark
uv run python -m pjepa_sim.verification.neural_active_boundary_claims
```

Generated files:

- `output/neural_active_boundary_benchmark.json`
- `output/neural_active_boundary_benchmark.md`
- `output/neural_active_boundary_verification.json`

## Neural Active-Probing Seed Sweep

```bash
cd simulation
uv run python -m pjepa_sim.cli.neural_active_seed_sweep_benchmark
uv run python -m pjepa_sim.verification.neural_active_seed_sweep_claims
```

Generated files:

- `output/neural_active_seed_sweep_benchmark.json`
- `output/neural_active_seed_sweep_benchmark.md`
- `output/neural_active_seed_sweep_verification.json`

## Pixel Continuous-Control Benchmark

```bash
cd simulation
uv run python -m pjepa_sim.cli.pixel_continuous_benchmark
uv run python -m pjepa_sim.verification.pixel_continuous_claims
```

Generated files:

- `output/pixel_continuous_benchmark.json`
- `output/pixel_continuous_benchmark.md`
- `output/pixel_continuous_verification.json`

## Video Representation Surrogate Benchmark

```bash
cd simulation
uv run python -m pjepa_sim.cli.video_representation_benchmark
uv run python -m pjepa_sim.verification.video_representation_claims
```

Generated files:

- `output/video_representation_benchmark.json`
- `output/video_representation_benchmark.md`
- `output/video_representation_verification.json`

This is a local passive-video JEPA surrogate, not a benchmark against V-JEPA or a video foundation model.

## Formal Contract-Interface Benchmark

```bash
cd simulation
uv run python -m pjepa_sim.cli.formal_contract_benchmark
uv run python -m pjepa_sim.verification.formal_contract_claims
```

Generated files:

- `output/formal_contract_benchmark.json`
- `output/formal_contract_benchmark.md`
- `output/formal_contract_verification.json`

This benchmark exports finite contracts for verification backends and checks them locally. It does not execute Kona or Aleph; those systems would need to consume the exported contract artifact and return proof or counterexample results.

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

## Load-Bearing KTH Real-Video Smoke Test

```bash
cd simulation
uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download
uv run python -m pjepa_sim.verification.kth_sample_video_claims
uv run python -m pjepa_sim.verification.manifest_video_protocol_claims
```

Generated files:

- `output/kth_sample_video_benchmark.json`
- `output/kth_sample_video_benchmark.md`
- `output/kth_sample_video_verification.json`

Downloaded videos are written under `simulation/data/kth_samples/`, which is gitignored. This verifier is part of `verify_all` once the sample files have been downloaded. It is a real-video smoke test using official KTH sample AVI files, not the full KTH benchmark.

## Manifest-Based Full-Video Protocol

```bash
cd simulation
uv run python -m pjepa_sim.cli.prepare_video_manifest kth --video-root path/to/kth-videos --output output/kth_full_manifest.csv
uv run python -m pjepa_sim.cli.manifest_video_benchmark --manifest path/to/manifest.csv --video-root path/to/videos --output-name full_video_benchmark
```

The manifest must contain `path`, `label`, and `split` columns. For a serious full-video benchmark, it must also contain a `group`, `subject`, or `scene` column such as subject, source video, capture session, or scene. The runner rejects same-file train/test leakage and, by default, requires train/test groups to be disjoint. Use `--require-action-metadata` when the result is meant to support a P-JEPA action-grounding claim rather than ordinary action recognition. Use `--validate-only` to check a large manifest without decoding videos.

The KTH builder assumes the usual filename pattern, such as `person15_walking_d1_uncomp.avi`, and defaults to subjects `01-16` for train and `17-25` for test. The six-file KTH sample set intentionally fails this protocol because it contains only one subject; it remains a smoke test, not a full benchmark.

## Robot Manifest Protocol

```bash
cd simulation
uv run python -m pjepa_sim.cli.validate_robot_manifest --manifest path/to/robot_manifest.csv --data-root path/to/data --require-language --require-robot-metadata
uv run python -m pjepa_sim.verification.robot_manifest_protocol_claims
```

The robot manifest must contain `episode_id`, `task`, `split`, `group`, `observation_path`, `action_path`, `success`, and `unsafe` columns before the project can make a robot-policy or safety claim from it. The validator checks that train/test tasks match, group ids are disjoint, referenced files exist, actions are present, success labels are present, and unsafe-failure labels are present by default. `--allow-missing-unsafe` can be used only for non-safety runs.

## Evidence-Level Guard

```bash
cd simulation
uv run python -m pjepa_sim.verification.evidence_claims
```

Generated files:

- `output/evidence_verification.json`
- `output/evidence_matrix.json`
- `output/EVIDENCE_MATRIX.md`

The evidence matrix classifies each local verifier and explicitly records broad claims that are not established by the current repository, including scalable JEPA replacement, video foundation model, learned robot policy, real robot competence, end-to-end neural sheaf learning, and unique cohomology advantage.

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
./scripts/build-paper.sh
```

The build writes `paper/PAPER.pdf`. The PDF is generated and ignored by git.

## Generated Output Policy

`simulation/output/`, `simulation/data/`, and `paper/PAPER.pdf` are ignored by git. They are reproducible output or downloaded benchmark data, not source.

If reviewers need a result snapshot, regenerate the output directory with the commands above rather than relying on stale checked-in artifacts.

## Expected Warning

Some Meta-World/Gymnasium runs print observation-space warnings from the wrapped environment. They have been observed in successful runs and are not currently treated as verifier failures. The verifier scripts are the authority for pass or fail.

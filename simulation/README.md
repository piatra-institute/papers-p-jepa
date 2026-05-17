# P-JEPA Simulation

This directory contains the executable evidence for the P-JEPA paper. The code tests hidden-regime action worlds where visually similar situations differ in action consequences.

For conceptual background, see:

- `../docs/ARCHITECTURE.md`
- `../docs/IMPLEMENTATION.md`
- `../docs/SCIENTIFIC_CLAIMS.md`
- `../docs/REPRODUCIBILITY.md`

## Core Commands

Run all local claim checks and generate the audit summary:

```bash
uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download
uv run python -m pjepa_sim.cli.verify_all
```

This writes `output/claims_summary.json` and `output/CLAIMS_SUMMARY.md`. The first command downloads six official KTH sample AVI files into `data/kth_samples/`; the directory is gitignored, but the KTH real-video verifier is part of the local audit.

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

Run the neural intervention encoder benchmark:

```bash
uv run python -m pjepa_sim.cli.neural_benchmark
uv run python -m pjepa_sim.verification.neural_claims
```

Run the neural intervention sample-efficiency benchmark:

```bash
uv run python -m pjepa_sim.cli.neural_sample_efficiency_benchmark
uv run python -m pjepa_sim.verification.neural_sample_efficiency_claims
```

Run the neural active-probing benchmark:

```bash
uv run python -m pjepa_sim.cli.neural_active_probe_benchmark
uv run python -m pjepa_sim.verification.neural_active_probe_claims
```

Run the neural active-probing boundary benchmark:

```bash
uv run python -m pjepa_sim.cli.neural_active_boundary_benchmark
uv run python -m pjepa_sim.verification.neural_active_boundary_claims
```

Run the neural active-probing seed-sweep benchmark:

```bash
uv run python -m pjepa_sim.cli.neural_active_seed_sweep_benchmark
uv run python -m pjepa_sim.verification.neural_active_seed_sweep_claims
```

Run the pixel-observation continuous-control benchmark:

```bash
uv run python -m pjepa_sim.cli.pixel_continuous_benchmark
uv run python -m pjepa_sim.verification.pixel_continuous_claims
```

Run the video-representation surrogate benchmark:

```bash
uv run python -m pjepa_sim.cli.video_representation_benchmark
uv run python -m pjepa_sim.verification.video_representation_claims
```

Run the formal contract-interface benchmark:

```bash
uv run python -m pjepa_sim.cli.formal_contract_benchmark
uv run python -m pjepa_sim.verification.formal_contract_claims
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

Run the load-bearing KTH real-video smoke test:

```bash
uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download
uv run python -m pjepa_sim.verification.kth_sample_video_claims
uv run python -m pjepa_sim.verification.manifest_video_protocol_claims
```

Prepare and validate a full-video manifest:

```bash
uv run python -m pjepa_sim.cli.prepare_video_manifest kth --video-root path/to/kth-videos --output output/kth_full_manifest.csv
uv run python -m pjepa_sim.cli.manifest_video_benchmark --manifest output/kth_full_manifest.csv --video-root path/to/kth-videos --validate-only --require-action-metadata
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
│   ├── formal/                       finite contract export for verification adapters
│   ├── representation/               action-grounded representation benchmark
│   ├── real_video/                   KTH real-video smoke test and manifest protocol
│   ├── verification/                 executable claim checks
│   └── paths.py                      repository-local paths
├── pyproject.toml                    uv project metadata
├── uv.lock                           locked Python environment
└── output/                           generated, gitignored artifacts
```

## What Is Being Tested

The exact toy world uses one visible object class and four hidden regimes: `dry`, `soapy`, `cracked`, and `heavy`. The hidden regime changes the success and unsafe-failure probabilities of task actions.

The benchmark compares no-probe predictive baselines, posterior-entropy probing, pure obstruction reduction, viability-aware P-JEPA probing, and oracle hidden-regime access. One suite also separates the policy's noisy belief sections from the true world sections to test a minimal learned-model mismatch. The representation benchmark separately tests whether unlabeled action/probe fingerprints support downstream action choice better than unstable visual grouping. The neural intervention encoder benchmark replaces engineered fingerprints with a small NumPy MLP trained from sampled intervention records over low-dimensional physical sensor observations and test identities. The neural sample-efficiency benchmark varies intervention repeats while holding the context stream fixed. The neural active-probing benchmark makes the initial structured sensor observation ambiguous and tests whether learned safe probes improve action choice. The neural active-probing boundary benchmark tests the same mechanism when sensors are aliased or distinct, probes are informative or weak, and probes are cheap or costly. The neural active-probing seed sweep repeats the aliased-sensor setting over deterministic seeds to check robustness of the no-probe margin and safety reduction. The pixel continuous-control benchmark replaces structured sensors with rendered image observations and replaces discrete task actions with continuous 2D controller rollouts. The video-representation surrogate benchmark compares a passive JEPA-like next-frame predictor with an action-conditioned predicted-test representation under visual style shift; it is not an actual V-JEPA benchmark. The formal contract-interface benchmark exports finite safety, branch-safety, obstruction, score, and probe-budget contracts for verification backends; the repository includes a local exhaustive checker but does not report Kona or Aleph runs. The online cover-construction benchmark tests whether the same local regimes can be built incrementally from an unlabeled stream. The scaling benchmark varies the number of synthetic hidden action regimes from 4 to 32 while keeping visual labels low-cardinality and shifted. The gluing ablation tests whether learned restriction maps help when local action sections use incompatible action-coordinate frames. The skill-composition benchmark tests whether those action-grounded representations support two-step precondition/postcondition chains.

The Meta-World adapter repeats the same comparison around a scripted continuous-control `reach-v3` controller. It is a probe-selection and hidden-regime inference benchmark, not a learned robotics policy benchmark.

The KTH sample benchmark downloads and decodes six official KTH sample AVI files. It is a load-bearing real-video smoke test in the local audit, not the full KTH benchmark. On the current temporal-window split, static appearance is stronger than temporal motion, which is evidence that the sample split is appearance dominated, not evidence that P-JEPA beats video methods. `pjepa_sim.real_video.manifest_benchmark` is the next-step runner for full real-video datasets; it requires manifest-level train/test splits, rejects same-video leakage, requires explicit group metadata by default, and can require action/intervention metadata when a P-JEPA action-grounding claim is being made. `pjepa_sim.real_video.manifest_builders` currently includes a KTH filename parser; the six-file sample set correctly fails as a full benchmark because it lacks a train/test subject split.

## Output Policy

`output/` is generated and ignored by git. `data/` contains downloaded benchmark data and is also ignored. Regenerate both with the commands above before changing paper numbers.

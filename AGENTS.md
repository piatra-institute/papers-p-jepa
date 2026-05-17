# Agent Instructions

This repository is a paper plus executable benchmark suite. Treat the paper, `README.md`, `simulation/README.md`, and `docs/` as the source of truth. `chat.md` is archival and should not be used as an authoritative dependency for future work.

## Working Rules

- Use `uv run python -m pjepa_sim...` for simulation commands.
- Keep command entry points package-local unless the user explicitly asks for root-level compatibility wrappers.
- Do not commit generated benchmark artifacts from `simulation/output/`; the directory is ignored and should be regenerated.
- Do not commit downloaded benchmark data from `simulation/data/`; the directory is ignored. The KTH real-video verifier is load-bearing, so run `uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download` before `verify_all` when the sample AVI files are missing.
- Do not commit `paper/PAPER.pdf`; it is a generated artifact and should be rebuilt from `paper/PAPER.md`.
- If a numeric claim changes, update all of these together: `paper/PAPER.md`, relevant verifier script, docs, and regenerated outputs.
- Do not add cross-references to other PIATRA papers as evidence. Lessons can be retained only if the paper stands on external literature and executable results.

## Verification Commands

From `simulation/`:

```bash
uv run python -m compileall -q pjepa_sim
uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download
uv run python -m pjepa_sim.cli.verify_all
uv run python -m pjepa_sim.verification.benchmark_claims
uv run python -m pjepa_sim.verification.representation_claims
uv run python -m pjepa_sim.verification.neural_claims
uv run python -m pjepa_sim.verification.neural_sample_efficiency_claims
uv run python -m pjepa_sim.verification.neural_active_probe_claims
uv run python -m pjepa_sim.verification.neural_active_boundary_claims
uv run python -m pjepa_sim.verification.neural_active_seed_sweep_claims
uv run python -m pjepa_sim.verification.pixel_continuous_claims
uv run python -m pjepa_sim.verification.video_representation_claims
uv run python -m pjepa_sim.verification.kth_sample_video_claims
uv run python -m pjepa_sim.verification.manifest_video_protocol_claims
uv run python -m pjepa_sim.verification.robot_manifest_protocol_claims
uv run python -m pjepa_sim.verification.formal_contract_claims
uv run python -m pjepa_sim.verification.online_claims
uv run python -m pjepa_sim.verification.scaling_claims
uv run python -m pjepa_sim.verification.gluing_claims
uv run python -m pjepa_sim.verification.composition_claims
uv run python -m pjepa_sim.verification.evidence_claims
uv run python -m pjepa_sim.verification.external_claims
uv run python -m pjepa_sim.verification.learned_external_claims
uv run python -m pjepa_sim.verification.unsupervised_external_claims
uv run python -m pjepa_sim.verification.stream_external_claims
uv run python -m pjepa_sim.verification.raw_record_external_claims
```

Full-video manifest preparation:

```bash
uv run python -m pjepa_sim.cli.prepare_video_manifest kth --video-root path/to/kth-videos --output output/kth_full_manifest.csv
uv run python -m pjepa_sim.cli.manifest_video_benchmark --manifest output/kth_full_manifest.csv --video-root path/to/kth-videos --validate-only --require-action-metadata
```

Robot/action manifest validation:

```bash
uv run python -m pjepa_sim.cli.validate_robot_manifest --manifest path/to/robot_manifest.csv --data-root path/to/data --require-language --require-robot-metadata
```

Optional external benchmark regeneration:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-benchmark --episodes 100
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-learned-benchmark --episodes 100 --train-probe-samples 64 --train-action-samples 256
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-unsupervised-benchmark --episodes 100 --unsupervised-contexts 32 --unsupervised-probe-trials 16 --unsupervised-action-trials 64
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-stream-benchmark --episodes 100 --stream-contexts 160 --unsupervised-probe-trials 16 --unsupervised-action-trials 64
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-raw-record-benchmark --episodes 100 --stream-contexts 160 --unsupervised-probe-trials 16 --unsupervised-action-trials 64
```

From the repository root:

```bash
./scripts/build-paper.sh
```

## Code Organization

- `simulation/pjepa_sim/core/`: exact toy environment, mathematical primitives, original simple agents, and plotting helpers.
- `simulation/pjepa_sim/benchmark/`: suite-level exact evaluator, entropy ablation, sheaf probe, and full P-JEPA stack.
- `simulation/pjepa_sim/representation/`: action-grounded representation, neural intervention encoder, neural active probing, online cover-construction, scaling, gluing-ablation, and skill-composition benchmarks.
- `simulation/pjepa_sim/perception/`: rendered-image and local continuous-control validity tests.
- `simulation/pjepa_sim/real_video/`: load-bearing KTH sample real-video smoke test and manifest-based full-video benchmark protocol using downloaded AVI files.
- `simulation/pjepa_sim/robot/`: robot/action manifest protocol for future robot-policy-learning evidence.
- `simulation/pjepa_sim/formal/`: finite contract export for external verification adapters.
- `simulation/pjepa_sim/external/`: Meta-World hidden-regime adapter and learned local-section estimators.
- `simulation/pjepa_sim/verification/`: executable claim checks and shared verifier reporting helpers.
- `simulation/pjepa_sim/cli/`: implementation for command-line entry points.
- `simulation/pjepa_sim/benchmark/configs/`: benchmark suite definitions loaded by the package.

Keep documentation in `docs/` current when architecture or scientific claims change.

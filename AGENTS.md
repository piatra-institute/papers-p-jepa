# Agent Instructions

This repository is a paper plus executable benchmark suite. Treat the paper, `README.md`, `simulation/README.md`, and `docs/` as the source of truth. `chat.md` is archival and should not be used as an authoritative dependency for future work.

## Working Rules

- Use `uv run python -m pjepa_sim...` for simulation commands.
- Keep command entry points package-local unless the user explicitly asks for root-level compatibility wrappers.
- Do not commit generated benchmark artifacts from `simulation/output/`; the directory is ignored and should be regenerated.
- Do not commit `paper/PAPER.pdf`; it is a generated artifact and should be rebuilt from `paper/PAPER.md`.
- If a numeric claim changes, update all of these together: `paper/PAPER.md`, relevant verifier script, docs, and regenerated outputs.
- Do not add cross-references to other PIATRA papers as evidence. Lessons can be retained only if the paper stands on external literature and executable results.

## Verification Commands

From `simulation/`:

```bash
uv run python -m compileall -q pjepa_sim
uv run python -m pjepa_sim.verification.benchmark_claims
uv run python -m pjepa_sim.verification.representation_claims
uv run python -m pjepa_sim.verification.online_claims
uv run python -m pjepa_sim.verification.scaling_claims
uv run python -m pjepa_sim.verification.gluing_claims
uv run python -m pjepa_sim.verification.composition_claims
uv run python -m pjepa_sim.verification.external_claims
uv run python -m pjepa_sim.verification.learned_external_claims
uv run python -m pjepa_sim.verification.unsupervised_external_claims
uv run python -m pjepa_sim.verification.stream_external_claims
uv run python -m pjepa_sim.verification.raw_record_external_claims
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
./paper/scripts/build-paper.sh
```

## Code Organization

- `simulation/pjepa_sim/core/`: exact toy environment, mathematical primitives, original simple agents, and plotting helpers.
- `simulation/pjepa_sim/benchmark/`: suite-level exact evaluator, entropy ablation, sheaf probe, and full P-JEPA stack.
- `simulation/pjepa_sim/representation/`: action-grounded representation, online cover-construction, scaling, gluing-ablation, and skill-composition benchmarks.
- `simulation/pjepa_sim/external/`: Meta-World hidden-regime adapter and learned local-section estimators.
- `simulation/pjepa_sim/verification/`: executable claim checks and shared verifier reporting helpers.
- `simulation/pjepa_sim/cli/`: implementation for command-line entry points.
- `simulation/pjepa_sim/benchmark/configs/`: benchmark suite definitions loaded by the package.

Keep documentation in `docs/` current when architecture or scientific claims change.

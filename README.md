# P-JEPA

P-JEPA is a paper and executable benchmark suite for a narrow claim about embodied predictive representations: when visually similar situations have different action consequences, an agent should learn local action models, measure when those models fail to agree, and use safe probes to repair the representation before acting.

The repository contains:

- `paper/PAPER.md`: the paper source.
- `simulation/`: exact hidden-regime benchmarks, Meta-World adapters, learned local-section models, and executable claim checks.
- `docs/`: durable project documentation replacing the exploratory chat log as the source of truth.

`chat.md` is archival context only. The project should be understandable from the paper, this README, `AGENTS.md`, and `docs/`.

## Quick Start

Run from `simulation/`:

```bash
uv run python -m pjepa_sim.cli.run_all
uv run python -m pjepa_sim.cli.benchmark --suite all --agents all
uv run python -m pjepa_sim.verification.benchmark_claims
uv run python -m pjepa_sim.cli.representation_benchmark
uv run python -m pjepa_sim.verification.representation_claims
uv run python -m pjepa_sim.cli.online_cover_benchmark
uv run python -m pjepa_sim.verification.online_claims
uv run python -m pjepa_sim.cli.scaling_benchmark
uv run python -m pjepa_sim.verification.scaling_claims
uv run python -m pjepa_sim.cli.gluing_ablation_benchmark
uv run python -m pjepa_sim.verification.gluing_claims
uv run python -m pjepa_sim.cli.skill_composition_benchmark
uv run python -m pjepa_sim.verification.composition_claims
```

Optional Meta-World runs require `gymnasium`, `metaworld`, and MuJoCo:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-raw-record-benchmark --episodes 100 --stream-contexts 160 --unsupervised-probe-trials 16 --unsupervised-action-trials 64
uv run python -m pjepa_sim.verification.raw_record_external_claims
```

Rebuild the paper from the repository root:

```bash
./paper/scripts/build-paper.sh
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md): conceptual and code-level structure.
- [Implementation](docs/IMPLEMENTATION.md): module map and extension points.
- [Scientific Claims](docs/SCIENTIFIC_CLAIMS.md): what is demonstrated, what is not demonstrated, and which verifier checks each claim.
- [Reproducibility](docs/REPRODUCIBILITY.md): commands, dependencies, generated artifacts, and expected outputs.

## Artifact Policy

`simulation/output/` and `paper/PAPER.pdf` are generated and gitignored. Regenerate them with the commands in `simulation/README.md`, `docs/REPRODUCIBILITY.md`, or the paper build command above.

If a numeric result changes, update the simulation output, verifiers, `paper/PAPER.md`, and rebuild `paper/PAPER.pdf` together.

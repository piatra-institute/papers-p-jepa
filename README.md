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
uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download
uv run python -m pjepa_sim.cli.verify_all
uv run python -m pjepa_sim.cli.action_grounding_challenge
uv run python -m pjepa_sim.verification.action_grounding_challenge_claims
uv run python -m pjepa_sim.cli.run_all
uv run python -m pjepa_sim.cli.benchmark --suite all --agents all
uv run python -m pjepa_sim.verification.benchmark_claims
uv run python -m pjepa_sim.cli.representation_benchmark
uv run python -m pjepa_sim.verification.representation_claims
uv run python -m pjepa_sim.cli.neural_benchmark
uv run python -m pjepa_sim.verification.neural_claims
uv run python -m pjepa_sim.cli.neural_sample_efficiency_benchmark
uv run python -m pjepa_sim.verification.neural_sample_efficiency_claims
uv run python -m pjepa_sim.cli.neural_active_probe_benchmark
uv run python -m pjepa_sim.verification.neural_active_probe_claims
uv run python -m pjepa_sim.cli.neural_active_boundary_benchmark
uv run python -m pjepa_sim.verification.neural_active_boundary_claims
uv run python -m pjepa_sim.cli.neural_active_seed_sweep_benchmark
uv run python -m pjepa_sim.verification.neural_active_seed_sweep_claims
uv run python -m pjepa_sim.cli.pixel_continuous_benchmark
uv run python -m pjepa_sim.verification.pixel_continuous_claims
uv run python -m pjepa_sim.cli.video_representation_benchmark
uv run python -m pjepa_sim.verification.video_representation_claims
uv run python -m pjepa_sim.verification.kth_sample_video_claims
uv run python -m pjepa_sim.verification.manifest_video_protocol_claims
uv run python -m pjepa_sim.verification.robot_manifest_protocol_claims
uv run python -m pjepa_sim.cli.formal_contract_benchmark
uv run python -m pjepa_sim.verification.formal_contract_claims
uv run python -m pjepa_sim.cli.online_cover_benchmark
uv run python -m pjepa_sim.verification.online_claims
uv run python -m pjepa_sim.cli.scaling_benchmark
uv run python -m pjepa_sim.verification.scaling_claims
uv run python -m pjepa_sim.cli.gluing_ablation_benchmark
uv run python -m pjepa_sim.verification.gluing_claims
uv run python -m pjepa_sim.cli.skill_composition_benchmark
uv run python -m pjepa_sim.verification.composition_claims
uv run python -m pjepa_sim.verification.evidence_claims
```

The action-grounding challenge is the current practical-use harness. It bundles the strongest local tests into one report: passive-representation failure, learned predicted-test representation, safe probe repair, learned restriction-map gluing, and skill composition.

The KTH command downloads six official sample AVI files into `simulation/data/kth_samples/`. That directory is gitignored, but the KTH verifier is part of the local audit; `verify_all` expects those files to be present.

Prepare a full KTH-style real-video manifest when the complete dataset is available:

```bash
uv run python -m pjepa_sim.cli.prepare_video_manifest kth --video-root path/to/kth-videos --output output/kth_full_manifest.csv
uv run python -m pjepa_sim.cli.manifest_video_benchmark --manifest output/kth_full_manifest.csv --video-root path/to/kth-videos --validate-only --require-action-metadata
```

Validate a future robot/action dataset manifest before making robot-policy claims:

```bash
uv run python -m pjepa_sim.cli.validate_robot_manifest --manifest path/to/robot_manifest.csv --data-root path/to/data --require-language --require-robot-metadata
```

Optional Meta-World runs require `gymnasium`, `metaworld`, and MuJoCo:

```bash
uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-raw-record-benchmark --episodes 100 --stream-contexts 160 --unsupervised-probe-trials 16 --unsupervised-action-trials 64
uv run python -m pjepa_sim.verification.raw_record_external_claims
```

Rebuild the paper from the repository root:

```bash
./scripts/build-paper.sh
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md): conceptual and code-level structure.
- [Implementation](docs/IMPLEMENTATION.md): module map and extension points.
- [Action-Grounding Challenge](docs/ACTION_GROUNDING_CHALLENGE.md): practical-use benchmark for passive-representation failure, predicted-test learning, probe repair, gluing, and composition.
- [Scientific Claims](docs/SCIENTIFIC_CLAIMS.md): what is demonstrated, what is not demonstrated, and which verifier checks each claim.
- [Claim Ledger](docs/CLAIM_LEDGER.md): reviewer-facing map from claims to executable evidence and limits.
- [Reproducibility](docs/REPRODUCIBILITY.md): commands, dependencies, generated artifacts, and expected outputs.
- [Formal Verification Adapters](docs/FORMAL_VERIFICATION_ADAPTERS.md): how to connect the finite contract export to external proof or constraint systems such as Kona or Aleph.
- [Next Validity Tests](docs/NEXT_VALIDITY_TESTS.md): the concrete benchmark ladder for proving or falsifying P-JEPA beyond the current toy and smoke-test evidence.

## Artifact Policy

`simulation/output/`, `simulation/data/`, and `paper/PAPER.pdf` are generated or downloaded artifacts and are gitignored. Regenerate them with the commands in `simulation/README.md`, `docs/REPRODUCIBILITY.md`, or the paper build command above.

If a numeric result changes, update the simulation output, verifiers, `paper/PAPER.md`, and rebuild `paper/PAPER.pdf` together.

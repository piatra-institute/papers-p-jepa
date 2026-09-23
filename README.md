# P-JEPA

JEPA Augmentations from Embodied and Causal Mathematics.

P-JEPA was proposed as a replacement for the homogeneous target embedding of Joint Embedding Predictive Architectures (JEPA): a sheaf-valued predictive state over a stratified interaction space. Its reference implementation computes a posterior-weighted variance in place of a coboundary and never places the sheaf in a training loop. We recast each piece of the proposal's embodied and causal mathematics (intervention prediction, bisimulation, active masking, viability, sheaf consistency, composition consistency) as an auxiliary loss, head, or sampler that plugs into a stock JEPA training loop and can be ablated. Five preregistered hypothesis tests on the dishworld toy and the reference code give the following results. The obstruction gate never fires on any of the five reported suites, so the full P-JEPA agent is numerically identical to a plain value-of-information agent. Value-aware active probing beats entropy probing across 50 seeds (paired bootstrap CI95 [+0.009, +0.017]). A frozen random projection of equal width matches the trained intervention encoder (CI95 [+0.000, +0.0045]). A genuine cellular sheaf reduces coboundary energy tenfold but chooses actions slightly worse than the raw cover (CI95 [−0.005, −0.004]). On a NumPy JEPA with toggleable auxiliary losses, the viability head shows a positive trend (CI95 [−0.007, +0.10]) and bisimulation at weight 0.3 hurts (CI95 [−0.23, −0.03]). The toy is at its variance limit, so these are directional signals. A typology mapping each augmentation to the data structure it suits, and a priority order for V-JEPA-scale ablation, are offered as conjecture; the toy partly contradicts the order, finding the first-ranked intervention loss neutral and the last-ranked viability head the only positive trend.

The repository contains:

- `paper/PAPER.md`: the paper source.
- `simulation/`: hidden-regime benchmarks, the NumPy JEPA toy, the preregistered hypothesis experiments, Meta-World adapters, and executable claim checks.
- `docs/`: project documentation, including `HYPOTHESIS_RESULTS.md` and `JEPA_AUGMENTATIONS.md`.

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

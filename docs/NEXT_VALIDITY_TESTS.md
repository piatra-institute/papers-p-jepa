# Next Validity Tests

This document turns the paper's current limits into executable milestones. The current repository supports a narrow P-JEPA claim under controlled hidden-regime, learned-probe, rendered-pixel, real-video smoke-test, and protocol-check settings. It does not yet prove that P-JEPA is a scalable JEPA replacement, a robot policy learner, or a video foundation model.

The generated `simulation/output/EVIDENCE_MATRIX.md` is the current claim boundary. It should change only after a new benchmark supplies performance evidence rather than protocol-only infrastructure.

## Milestone 1: Full Real-Video Benchmark

Question: does the action-grounding argument survive a real video dataset with leakage-aware train/test splits?

Use `simulation/pjepa_sim/real_video/manifest_benchmark.py` through:

```bash
cd simulation
uv run python -m pjepa_sim.cli.prepare_video_manifest kth --video-root path/to/kth-videos --output output/kth_full_manifest.csv
uv run python -m pjepa_sim.cli.manifest_video_benchmark --manifest path/to/manifest.csv --video-root path/to/videos --output-name full_video_benchmark
```

The manifest must contain `path`, `label`, and `split`. For a serious result it must also contain `group`, `subject`, or `scene`, where the group is subject, scene, source video, or capture session. The runner rejects same-file train/test leakage and requires train/test group disjointness by default. Add `--require-action-metadata` when the claim is specifically about P-JEPA action grounding rather than ordinary video action recognition. Add `--validate-only` for large manifests when you want protocol checks before decoding video.

Candidate datasets:

- Something-Something V2 for fine-grained object-action video.
- Ego4D or EgoSchema-derived clips for egocentric temporal understanding.
- Full KTH or another action-recognition dataset with subject-disjoint splits as a low-cost first pass.
- DROID robot videos when action/intervention metadata is part of the manifest.

Pass condition: the result must use real files, disjoint groups, matched train/test classes, and declared baselines. A P-JEPA claim additionally requires action/intervention metadata or a defensible proxy for action tests.

## Milestone 2: Compute-Matched JEPA Comparison

Question: can a P-JEPA objective match ordinary JEPA on standard video representation while improving action-relevant transfer?

Required implementation:

- PyTorch video encoder.
- Passive JEPA baseline trained under the same data, augmentations, compute, and model size.
- P-JEPA objective with latent prediction, action/test-conditioned prediction, restriction consistency, obstruction calibration, and viability where the dataset supports it.
- Frozen-probe and fine-tuning evaluation.

Pass condition: P-JEPA must not collapse ordinary video accuracy while improving at least one action- or intervention-relevant downstream metric under matched compute.

## Milestone 3: Robot Policy Learning

Question: does P-JEPA improve policies, not only representations?

Candidate benchmarks:

- LIBERO for language-conditioned simulated manipulation.
- RoboMimic for offline imitation.
- Meta-World or ManiSkill for controlled continuous-control comparisons.
- DROID or Open X-Embodiment for real robot-video/action data.

Before a robot benchmark is treated as evidence, validate its manifest:

```bash
cd simulation
uv run python -m pjepa_sim.cli.validate_robot_manifest --manifest path/to/robot_manifest.csv --data-root path/to/data --require-language --require-robot-metadata
```

The manifest must expose observation data, action trajectories, task labels, train/test groups, success metrics, and unsafe-failure metrics. Without these fields the result may still be a useful dataset experiment, but it cannot support the paper's robot-policy or safety claims.

Required baselines:

- Behavioral cloning.
- Diffusion policy or ACT where available.
- Frozen passive video encoder plus policy head.
- P-JEPA encoder plus the same policy head.
- P-JEPA ablations: no gluing, no obstruction, no action tests, no viability.

Pass condition: P-JEPA improves task success, unsafe failure, OOD recovery, data efficiency, or perturbation recovery under the same policy learner and training budget.

## Milestone 4: Scaling Evidence

Question: does the objective scale?

Required sweeps:

- Data: at least three dataset sizes.
- Model: at least three parameter scales.
- Compute: matched training schedules for passive JEPA and P-JEPA.
- Downstream: at least one video metric and one robot/action metric.

Pass condition: the P-JEPA transfer curve must improve with scale and must be competitive with passive JEPA on general video representation. Without this, P-JEPA remains a mechanism and benchmark idea, not a foundation-model replacement.

## Milestone 5: Real Intervention Evidence

Question: does active probing repair representation under real or high-fidelity physical uncertainty?

Required evidence:

- Hidden but action-relevant condition such as friction, mass, compliance, grasp stability, or occluded affordance.
- Safe probes with measurable cost.
- Explicit unsafe-failure metric.
- A no-probe baseline and an entropy-probe baseline.
- A held-out regime or held-out scene split.

Pass condition: value-aware P-JEPA probing improves safety or score relative to no probing and entropy probing, and the improvement persists under held-out regimes or scenes.

## Current Priority

The next concrete step is Milestone 1 on a full real-video dataset. The repository now has the manifest runner and KTH filename parser needed to do this without repeating the KTH sample leakage problem. The six-file KTH sample intentionally fails as a full split; a full KTH run requires the complete subject set, and a P-JEPA action-grounding run requires action/intervention metadata. The result should be reported even if it is negative; a negative full-video result would be scientifically useful because it would identify whether the current P-JEPA evidence is still confined to controlled intervention benchmarks.

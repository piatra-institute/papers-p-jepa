# Claim Ledger

This document is the durable reviewer-facing map for the executable claims. The generated version lives at `simulation/output/CLAIMS_SUMMARY.md` and is rebuilt by:

```bash
cd simulation
uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download
uv run python -m pjepa_sim.cli.verify_all
```

The generated summary is intentionally not committed. It records every local verifier claim, observed value, threshold, verifier JSON, and limitation.

## Local Claims

The local verification audit covers these evidence groups:

- Exact hidden-regime mechanism: obstruction exists, safe probes reduce it, and action choice changes in hidden-risk regimes.
- Suite-level P-JEPA stack: viability-aware probing beats prior, pure obstruction reduction, and entropy probing across configured toy suites.
- Action-grounded representation: action/probe fingerprints beat unstable visual grouping under visual cue shift.
- Neural intervention encoder: a small NumPy MLP learns predicted-test vectors from structured sensor observations and intervention records.
- Neural sample efficiency: the learned predicted-test vector remains useful across sparse intervention-repeat budgets in the toy setting.
- Neural active probing: learned value-aware probing repairs aliased structured observations before action.
- Neural active-probing boundary: the active-probing gain weakens when sensors already identify regimes or probes are weak, and costly probes reduce probe use.
- Neural active-probing seed sweep: the no-probe margin and unsafe-failure reduction persist across tested deterministic seeds.
- Pixel continuous control: rendered pixel observations and continuous 2D controller rollouts retain a modest active-probing advantage.
- Video representation surrogate: a passive JEPA-like next-frame predictor can predict frames while failing to recover action regimes under visual shift, whereas action-conditioned predicted-test representation recovers the action regimes.
- KTH sample real video: the load-bearing real-video smoke test uses downloaded official KTH sample AVI files and currently gives a diagnostic negative result for P-JEPA video advantage because static appearance and passive next-frame descriptors beat temporal motion on the sample split.
- Manifest real-video protocol: the audit checks that the next full-video benchmark runner rejects class-incomplete splits, same-file train/test leakage, missing group metadata, and missing action metadata when a P-JEPA action-grounding claim is requested. It also checks that the KTH manifest builder parses official-style filenames while rejecting the six-file sample set as a full benchmark.
- Formal contract interface: finite safety, branch-safety, obstruction, score, and probe-budget contracts are exported for verification backends and checked locally without claiming Kona or Aleph execution.
- Online cover construction: the action-regime cover can be built incrementally from engineered fingerprints.
- Synthetic regime scaling: engineered action-consequence grouping remains coherent as synthetic hidden-regime count increases.
- Restriction-map gluing: learned linear restriction maps align incompatible local action-coordinate frames.
- Skill composition: action-grounded representations select the intended two-step precondition/postcondition chains.

## Outside The Local Audit

Optional Meta-World adapter checks are not part of `verify_all` because they require `gymnasium`, `metaworld`, and MuJoCo runtime dependencies. They remain documented in `simulation/README.md` and `docs/REPRODUCIBILITY.md`.

## Standing Limits

The local audit does not prove robot competence, tactile-stream representation learning, internet-scale video representation learning, or end-to-end neural sheaf learning. The pixel continuous-control benchmark is only a small rendered-image stress test; it does not establish real-world vision or robot control. The video-representation benchmark is a local surrogate for passive JEPA-style prediction, not a result against V-JEPA, V-JEPA 2, or any video foundation model. The KTH sample check is load-bearing but small; it proves the audit can process real video and records the current negative sample result, not full action-video superiority. The manifest protocol check is infrastructure, not performance evidence. The formal contract interface is a local finite-state checker and export protocol, not a benchmark result for proprietary Kona or Aleph systems. The audit proves that the repository's controlled executable claims still pass under their stated assumptions.

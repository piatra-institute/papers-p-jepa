# Action-Grounding Challenge

The action-grounding challenge is the current practical-use benchmark for P-JEPA. It answers a narrow question: when passive or appearance-based representations are misleading, can an action-conditioned representation support safer action, useful probes, local-model alignment, and simple skill composition?

Run it from `simulation/`:

```bash
uv run python -m pjepa_sim.cli.action_grounding_challenge
uv run python -m pjepa_sim.verification.action_grounding_challenge_claims
```

Generated files:

- `output/action_grounding_challenge.json`
- `output/action_grounding_challenge.md`
- `output/action_grounding_challenge_verification.json`

## What It Tests

The challenge bundles five controlled local tests into one report.

| Step | Question | Required result |
|---|---|---|
| Passive failure | Can passive prediction be accurate while action-regime transfer fails? | Action-conditioned score beats passive video by more than `0.30`, while passive prediction MAE remains below `0.04` and passive action-regime purity remains below `0.70`. |
| Action-consequence transfer | Does action evidence beat visual grouping under cue shift? | Action-consequence grouping beats appearance by more than `0.25` and prior averaging by more than `0.10`. |
| Predicted-test learning | Can the representation be learned from intervention records? | Neural predicted-test representation beats appearance by more than `0.20`, reaches purity above `0.85`, and uses no hidden-label features. |
| Probe repair | Can safe probes repair an ambiguous representation before action? | Active probing beats no probing by more than `0.18`, reduces unsafe failure by more than `0.06`, and uses more than `0.50` probes on average. |
| Local gluing | Do restriction maps help when local action sections use incompatible coordinates? | Learned gluing reduces overlap residual below `0.10` of identity/no-glue and improves score by more than `0.15`. |
| Composition | Does the representation support skill-chain selection? | Action-grounded composition beats appearance and prior baselines, reaches purity above `0.95`, and selects all expected chains. |

The final check requires all component checks to pass together.

## Current Result

The current local audit passes the challenge. The generated report records:

| Step | Key metric | Current value |
|---|---|---:|
| Passive failure | `p_action_minus_passive_video_score` | `0.520` |
| Predicted-test learning | `neural_p_minus_appearance_score` | `0.520` |
| Probe repair | `active_minus_no_probe_score` | `0.213` |
| Local gluing | `learned_glue_minus_identity_score` | `0.224` |
| Composition | `composition_minus_appearance_score` | `0.390` |

These numbers come from `simulation/output/action_grounding_challenge.md` after regeneration.

## What It Does Not Prove

The challenge is an integrated controlled harness. It does not prove real robot competence, real-video advantage, foundation-model scaling, tactile-stream representation learning, or end-to-end neural sheaf learning. It is useful because it turns the current practical P-JEPA claim into a single executable benchmark with explicit thresholds and limitations.

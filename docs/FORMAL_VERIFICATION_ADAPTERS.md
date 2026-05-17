# Formal Verification Adapters

P-JEPA is a learning mechanism, not a proof system. The clean way to compare or integrate it with systems such as Kona or Aleph is to export the learned policy interface as a finite contract and ask an external verifier to prove the contract or return counterexamples.

The current repository includes a local deterministic checker in `simulation/pjepa_sim/formal/contracts.py`. It is deliberately modest: it checks finite hidden-regime benchmark suites by exact enumeration. It does not execute Kona, Aleph, Lean, or any proprietary verification backend.

## Contract Artifact

Run:

```bash
cd simulation
uv run python -m pjepa_sim.cli.formal_contract_benchmark
```

This writes:

- `output/formal_contract_benchmark.json`
- `output/formal_contract_benchmark.md`

The JSON artifact contains:

- benchmark suite name,
- agent name,
- observed policy metrics,
- contract requirements,
- pass/fail checks,
- counterexamples for violated requirements,
- an explicit `external_backends_executed` field.

## Current Contract Requirements

The local checker currently verifies:

- expected unsafe failure is bounded,
- worst hidden-regime branch unsafe failure is bounded,
- risk-adjusted score stays above a floor,
- residual obstruction at action stays below a ceiling,
- mean probe count stays inside the probe budget.

These are finite policy contracts, not general robotics safety theorems. They are useful because they separate the learning question from the certification question.

## How Kona Or Aleph Would Fit

A Kona-style constraint backend would consume the exported finite state/action model and evaluate whether the policy stays inside the allowed safety and viability region across all reachable hidden-regime branches.

An Aleph-style proof backend would consume the same artifact after translation into a formal language such as Lean, prove the stated requirements, or return a proof failure and counterexample. The benchmark score would then include proof success, proof time, counterexample quality, and any mismatch between the exported model and the intended physical semantics.

## What Can Be Claimed Now

The current claim is only:

```text
P-JEPA policies can be exported into finite verification contracts, and a local exhaustive checker distinguishes P-JEPA from weaker baselines under the configured safety-efficiency budget.
```

The current claim is not:

```text
Kona or Aleph has been benchmarked on P-JEPA.
```

That claim requires actual access to the external system, a stable input schema, a recorded backend version, and generated proof or counterexample artifacts.


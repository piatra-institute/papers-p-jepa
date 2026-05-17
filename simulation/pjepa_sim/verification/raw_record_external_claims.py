"""Verify the raw-record Meta-World benchmark claims."""

from __future__ import annotations

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.verification.reporting import load_json, strategy_aggregates, write_margin_report


OUT = OUTPUT_DIR
RESULTS = OUT / "metaworld_raw_record_strategy_benchmark.json"
VERIFY = OUT / "metaworld_raw_record_strategy_verification.json"


def main() -> int:
    results = load_json(RESULTS)

    diagnostics = results["fit_diagnostics"]
    training = results["training"]
    agents = strategy_aggregates(results, nested=True)
    raw_counts = training["raw_dataset"]["contexts_per_true_regime"]
    derived_counts = training["derived_dataset"]["fingerprints_per_true_regime"]
    checks = {
        "raw_record_saw_every_regime": min(raw_counts.values()) - 1,
        "raw_record_preserved_context_counts": min(
            derived_counts[regime] - raw_counts[regime]
            for regime in raw_counts
        ),
        "raw_record_cluster_purity_above_0_95": diagnostics["cluster_purity"] - 0.95,
        "raw_record_probe_likelihood_mae_below_0_08": 0.08 - diagnostics["probe_likelihood_mae"],
        "raw_record_local_section_mae_below_0_08": 0.08 - diagnostics["section_mae"],
        "raw_record_obstruction_probe_reduces_unsafe_vs_no_probe": (
            agents["fixed_no_probe"]["unsafe_rate"]
            - agents["obstruction_probe_belief_safe"]["unsafe_rate"]
        ),
        "raw_record_obstruction_probe_beats_no_probe_score": (
            agents["obstruction_probe_belief_safe"]["risk_adjusted_score"]
            - agents["fixed_no_probe"]["risk_adjusted_score"]
        ),
        "raw_record_obstruction_probe_beats_same_budget_random_unsafe": (
            agents["random_one_probe_belief_safe"]["unsafe_rate"]
            - agents["obstruction_probe_belief_safe"]["unsafe_rate"]
        ),
        "raw_record_obstruction_probe_beats_same_budget_random_score": (
            agents["obstruction_probe_belief_safe"]["risk_adjusted_score"]
            - agents["random_one_probe_belief_safe"]["risk_adjusted_score"]
        ),
        "raw_record_obstruction_probe_beats_entropy_unsafe": (
            agents["entropy_probe_belief_safe"]["unsafe_rate"]
            - agents["obstruction_probe_belief_safe"]["unsafe_rate"]
        ),
        "raw_record_obstruction_probe_beats_entropy_score": (
            agents["obstruction_probe_belief_safe"]["risk_adjusted_score"]
            - agents["entropy_probe_belief_safe"]["risk_adjusted_score"]
        ),
        "raw_record_obstruction_probe_uses_fewer_probes_than_exhaustive_random": (
            agents["random_probe_belief_safe"]["mean_probes"]
            - agents["obstruction_probe_belief_safe"]["mean_probes"]
        ),
        "raw_record_obstruction_probe_beats_exhaustive_random_score": (
            agents["obstruction_probe_belief_safe"]["risk_adjusted_score"]
            - agents["random_probe_belief_safe"]["risk_adjusted_score"]
        ),
    }
    return write_margin_report(RESULTS, VERIFY, checks, pass_on_zero=True)


if __name__ == "__main__":
    raise SystemExit(main())

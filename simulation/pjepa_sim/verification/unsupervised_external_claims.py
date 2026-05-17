"""Verify the unsupervised Meta-World hidden-regime benchmark claims."""

from __future__ import annotations

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.verification.reporting import load_json, strategy_aggregates, write_margin_report


OUT = OUTPUT_DIR
RESULTS = OUT / "metaworld_unsupervised_strategy_benchmark.json"
VERIFY = OUT / "metaworld_unsupervised_strategy_verification.json"


def main() -> int:
    results = load_json(RESULTS)

    diagnostics = results["fit_diagnostics"]
    agents = strategy_aggregates(results, nested=True)
    checks = {
        "unsupervised_cluster_purity_above_0_95": diagnostics["cluster_purity"] - 0.95,
        "unsupervised_probe_likelihood_mae_below_0_08": 0.08 - diagnostics["probe_likelihood_mae"],
        "unsupervised_local_section_mae_below_0_08": 0.08 - diagnostics["section_mae"],
        "unsupervised_obstruction_probe_reduces_unsafe_vs_no_probe": (
            agents["fixed_no_probe"]["unsafe_rate"]
            - agents["obstruction_probe_belief_safe"]["unsafe_rate"]
        ),
        "unsupervised_obstruction_probe_beats_no_probe_score": (
            agents["obstruction_probe_belief_safe"]["risk_adjusted_score"]
            - agents["fixed_no_probe"]["risk_adjusted_score"]
        ),
        "unsupervised_obstruction_probe_beats_same_budget_random_unsafe": (
            agents["random_one_probe_belief_safe"]["unsafe_rate"]
            - agents["obstruction_probe_belief_safe"]["unsafe_rate"]
        ),
        "unsupervised_obstruction_probe_beats_same_budget_random_score": (
            agents["obstruction_probe_belief_safe"]["risk_adjusted_score"]
            - agents["random_one_probe_belief_safe"]["risk_adjusted_score"]
        ),
        "unsupervised_obstruction_probe_beats_entropy_unsafe": (
            agents["entropy_probe_belief_safe"]["unsafe_rate"]
            - agents["obstruction_probe_belief_safe"]["unsafe_rate"]
        ),
        "unsupervised_obstruction_probe_beats_entropy_score": (
            agents["obstruction_probe_belief_safe"]["risk_adjusted_score"]
            - agents["entropy_probe_belief_safe"]["risk_adjusted_score"]
        ),
        "unsupervised_obstruction_probe_uses_fewer_probes_than_exhaustive_random": (
            agents["random_probe_belief_safe"]["mean_probes"]
            - agents["obstruction_probe_belief_safe"]["mean_probes"]
        ),
        "unsupervised_obstruction_probe_matches_exhaustive_random_score_within_tolerance": (
            0.001
            - abs(
                agents["obstruction_probe_belief_safe"]["risk_adjusted_score"]
                - agents["random_probe_belief_safe"]["risk_adjusted_score"]
            )
        ),
    }
    return write_margin_report(RESULTS, VERIFY, checks)


if __name__ == "__main__":
    raise SystemExit(main())

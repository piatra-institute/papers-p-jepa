"""External benchmark entry point for P-JEPA.

This command can always write the external protocol and dependency status. It
only runs Meta-World episodes when the optional Meta-World/MuJoCo stack is
importable.

Examples:

    uv run python -m pjepa_sim.cli.external_benchmark --check
    uv run python -m pjepa_sim.cli.external_benchmark --protocol
    uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-smoke --episodes 2
    uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-benchmark --episodes 20
    uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-learned-benchmark --episodes 20
    uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-unsupervised-benchmark --episodes 20
    uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-stream-benchmark --episodes 20
    uv run --with gymnasium --with metaworld python -m pjepa_sim.cli.external_benchmark --run-raw-record-benchmark --episodes 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pjepa_sim.external.metaworld_hidden_regime import (
    MetaWorldHiddenRegimeConfig,
    check_metaworld,
    protocol_summary,
    run_smoke,
    run_strategy_benchmark,
)
from pjepa_sim.external.learned_metaworld import (
    run_learned_strategy_benchmark,
    run_raw_record_strategy_benchmark,
    run_stream_unsupervised_strategy_benchmark,
    run_unsupervised_strategy_benchmark,
)
from pjepa_sim.paths import OUTPUT_DIR


OUT = OUTPUT_DIR


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    config = MetaWorldHiddenRegimeConfig(
        task=args.task,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        obstruction_threshold=args.obstruction_threshold,
        max_probes=args.max_probes,
    )
    status = check_metaworld()
    protocol = protocol_summary(config)
    payload = {
        "external_benchmarks": {
            "metaworld_hidden_regime": {
                "availability": status.as_dict(),
                "protocol": protocol,
            }
        }
    }

    requested_run = (
        args.run_smoke
        or args.run_benchmark
        or args.run_learned_benchmark
        or args.run_unsupervised_benchmark
        or args.run_stream_benchmark
        or args.run_raw_record_benchmark
    )
    if args.check or args.protocol or not requested_run:
        write_json(payload, OUT / "external_benchmark_status.json")
        print_status(payload)
        print(f"Wrote: {OUT / 'external_benchmark_status.json'}")

    if requested_run and not status.available:
        write_json(payload, OUT / "external_benchmark_status.json")
        print_status(payload)
        print()
        print("Meta-World is not importable, so no external episodes were run.")
        print(f"Install hint: {status.install_hint}")
        print(f"Wrote: {OUT / 'external_benchmark_status.json'}")
        return 2

    if args.run_smoke:
        smoke = run_smoke(
            config,
            episodes=args.episodes,
            policy_name=args.policy,
            probe_strategy=args.probe_strategy,
        )
        smoke_path = OUT / f"metaworld_{args.policy}_{args.probe_strategy}_smoke.json"
        write_json(smoke, smoke_path)
        print(f"Ran {args.episodes} Meta-World hidden-regime smoke episode(s).")
        print(f"Wrote: {smoke_path}")

    if args.run_benchmark:
        strategies = None if args.strategies == ["all"] else args.strategies
        benchmark = run_strategy_benchmark(config, episodes=args.episodes, strategy_names=strategies)
        benchmark_path = OUT / "metaworld_strategy_benchmark.json"
        table_path = OUT / "metaworld_strategy_benchmark.md"
        write_json(benchmark, benchmark_path)
        write_strategy_table(benchmark, table_path)
        print_strategy_table(benchmark)
        print(f"Wrote: {benchmark_path}")
        print(f"Wrote: {table_path}")

    if args.run_learned_benchmark:
        strategies = None if args.strategies == ["all"] else args.strategies
        learned = run_learned_strategy_benchmark(
            config=config,
            train_probe_samples_per_regime_probe=args.train_probe_samples,
            train_action_samples_per_regime=args.train_action_samples,
            eval_episodes=args.episodes,
            seed=args.seed + 10_000,
            strategy_names=strategies,
        )
        learned_path = OUT / "metaworld_learned_strategy_benchmark.json"
        table_path = OUT / "metaworld_learned_strategy_benchmark.md"
        write_json(learned, learned_path)
        write_strategy_table(learned["evaluation"], table_path)
        print_learned_summary(learned)
        print_strategy_table(learned["evaluation"])
        print(f"Wrote: {learned_path}")
        print(f"Wrote: {table_path}")

    if args.run_unsupervised_benchmark:
        strategies = None if args.strategies == ["all"] else args.strategies
        unsupervised = run_unsupervised_strategy_benchmark(
            config=config,
            contexts_per_regime=args.unsupervised_contexts,
            probe_trials_per_probe=args.unsupervised_probe_trials,
            action_trials_per_context=args.unsupervised_action_trials,
            eval_episodes=args.episodes,
            seed=args.seed + 20_000,
            strategy_names=strategies,
        )
        unsupervised_path = OUT / "metaworld_unsupervised_strategy_benchmark.json"
        table_path = OUT / "metaworld_unsupervised_strategy_benchmark.md"
        write_json(unsupervised, unsupervised_path)
        write_strategy_table(unsupervised["evaluation"], table_path)
        print_unsupervised_summary(unsupervised)
        print_strategy_table(unsupervised["evaluation"])
        print(f"Wrote: {unsupervised_path}")
        print(f"Wrote: {table_path}")

    if args.run_stream_benchmark:
        strategies = None if args.strategies == ["all"] else args.strategies
        stream = run_stream_unsupervised_strategy_benchmark(
            config=config,
            total_contexts=args.stream_contexts,
            probe_trials_per_probe=args.unsupervised_probe_trials,
            action_trials_per_context=args.unsupervised_action_trials,
            eval_episodes=args.episodes,
            seed=args.seed + 30_000,
            strategy_names=strategies,
        )
        stream_path = OUT / "metaworld_stream_unsupervised_strategy_benchmark.json"
        table_path = OUT / "metaworld_stream_unsupervised_strategy_benchmark.md"
        write_json(stream, stream_path)
        write_strategy_table(stream["evaluation"], table_path)
        print_stream_summary(stream)
        print_strategy_table(stream["evaluation"])
        print(f"Wrote: {stream_path}")
        print(f"Wrote: {table_path}")

    if args.run_raw_record_benchmark:
        strategies = None if args.strategies == ["all"] else args.strategies
        raw_record = run_raw_record_strategy_benchmark(
            config=config,
            total_contexts=args.stream_contexts,
            probe_trials_per_probe=args.unsupervised_probe_trials,
            action_trials_per_context=args.unsupervised_action_trials,
            eval_episodes=args.episodes,
            seed=args.seed + 40_000,
            strategy_names=strategies,
        )
        raw_record_path = OUT / "metaworld_raw_record_strategy_benchmark.json"
        table_path = OUT / "metaworld_raw_record_strategy_benchmark.md"
        write_json(raw_record, raw_record_path)
        write_strategy_table(raw_record["evaluation"], table_path)
        print_raw_record_summary(raw_record)
        print_strategy_table(raw_record["evaluation"])
        print(f"Wrote: {raw_record_path}")
        print(f"Wrote: {table_path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether optional external benchmark dependencies are importable.",
    )
    parser.add_argument(
        "--protocol",
        action="store_true",
        help="Write the Meta-World hidden-regime protocol without running episodes.",
    )
    parser.add_argument(
        "--run-smoke",
        "--run-random-smoke",
        dest="run_smoke",
        action="store_true",
        help="Run smoke episodes when Meta-World is installed.",
    )
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Compare external hidden-regime strategies in Meta-World.",
    )
    parser.add_argument(
        "--run-learned-benchmark",
        action="store_true",
        help="Fit a learned hidden-regime model, then compare external strategies.",
    )
    parser.add_argument(
        "--run-unsupervised-benchmark",
        action="store_true",
        help="Fit an unlabeled clustered hidden-regime model, then compare strategies.",
    )
    parser.add_argument(
        "--run-stream-benchmark",
        action="store_true",
        help="Fit an unlabeled prior-sampled stream model, then compare strategies.",
    )
    parser.add_argument(
        "--run-raw-record-benchmark",
        action="store_true",
        help="Fit an unlabeled raw-record model, then compare strategies.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes per smoke run or strategy.",
    )
    parser.add_argument(
        "--policy",
        choices=["reach_scripted", "reach_belief_safe", "random"],
        default="reach_scripted",
        help="Smoke-test policy. This is plumbing validation, not a learned baseline.",
    )
    parser.add_argument(
        "--probe-strategy",
        choices=["none", "random", "entropy", "obstruction", "oracle"],
        default="obstruction",
        help="Probe strategy for a smoke run.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["all"],
        help="Strategy names for --run-benchmark, or 'all'.",
    )
    parser.add_argument(
        "--task",
        default="reach-v3",
        help="Meta-World task name passed as env_name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the wrapper and environment.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=150,
        help="Maximum environment steps per external episode.",
    )
    parser.add_argument(
        "--obstruction-threshold",
        type=float,
        default=0.150,
        help="Obstruction level below which the active external strategy stops probing.",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=3,
        help="Maximum number of probes before acting.",
    )
    parser.add_argument(
        "--train-probe-samples",
        type=int,
        default=64,
        help="Training probe samples per hidden-regime/probe pair for learned benchmark.",
    )
    parser.add_argument(
        "--train-action-samples",
        type=int,
        default=256,
        help="Training action-effect samples per hidden regime for learned benchmark.",
    )
    parser.add_argument(
        "--unsupervised-contexts",
        type=int,
        default=32,
        help="Unlabeled context fingerprints per true regime for unsupervised benchmark.",
    )
    parser.add_argument(
        "--unsupervised-probe-trials",
        type=int,
        default=16,
        help="Probe trials per probe inside each unlabeled context fingerprint.",
    )
    parser.add_argument(
        "--unsupervised-action-trials",
        type=int,
        default=64,
        help="Action-effect trials inside each unlabeled context fingerprint.",
    )
    parser.add_argument(
        "--stream-contexts",
        type=int,
        default=160,
        help="Total prior-sampled unlabeled context fingerprints for stream benchmark.",
    )
    return parser.parse_args()


def write_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_strategy_table(benchmark: dict[str, Any], path: Path) -> None:
    lines = [
        "| Strategy | Success | Unsafe | Probes | Obstruction at action | Risk-adjusted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in benchmark["strategies"].items():
        metrics = result["aggregate"]
        lines.append(
            "| "
            f"`{name}` | "
            f"{metrics['success_rate']:.3f} | "
            f"{metrics['unsafe_rate']:.3f} | "
            f"{metrics['mean_probes']:.3f} | "
            f"{metrics['mean_obstruction_at_action']:.3f} | "
            f"{metrics['risk_adjusted_score']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def print_status(payload: dict[str, Any]) -> None:
    item = payload["external_benchmarks"]["metaworld_hidden_regime"]
    availability = item["availability"]
    protocol = item["protocol"]
    print("External benchmark status")
    print(f"  metaworld available: {availability['available']}")
    if availability["error"]:
        print(f"  import error: {availability['error']}")
    print(f"  install hint: {availability['install_hint']}")
    print(f"  API hint: {availability['api_hint']}")
    print()
    print("Meta-World hidden-regime protocol")
    print(f"  task: {protocol['config']['task']}")
    print(f"  regimes: {', '.join(protocol['config']['regimes'])}")
    print(f"  probes: {', '.join(protocol['config']['probes'])}")
    print(f"  prior obstruction: {protocol['obstruction_prior']:.6f}")
    if protocol["preflight_probe_order"]:
        order = [item["probe"] for item in protocol["preflight_probe_order"]]
        print(f"  preflight probe order: {', '.join(order)}")


def print_strategy_table(benchmark: dict[str, Any]) -> None:
    print("Meta-World hidden-regime strategy comparison")
    for name, result in benchmark["strategies"].items():
        metrics = result["aggregate"]
        print(
            f"  {name:>32s}  "
            f"success={metrics['success_rate']:.3f}  "
            f"unsafe={metrics['unsafe_rate']:.3f}  "
            f"probes={metrics['mean_probes']:.3f}  "
            f"O={metrics['mean_obstruction_at_action']:.3f}  "
            f"score={metrics['risk_adjusted_score']:.3f}"
        )


def print_learned_summary(learned: dict[str, Any]) -> None:
    diagnostics = learned["fit_diagnostics"]
    training = learned["training"]
    print("Learned Meta-World hidden-regime model")
    print(f"  probe samples per regime/probe: {training['probe_samples_per_regime_probe']}")
    print(f"  action samples per regime: {training['action_samples_per_regime']}")
    print(f"  probe likelihood MAE: {diagnostics['probe_likelihood_mae']:.4f}")
    print(f"  local section MAE: {diagnostics['section_mae']:.4f}")
    print()


def print_unsupervised_summary(unsupervised: dict[str, Any]) -> None:
    diagnostics = unsupervised["fit_diagnostics"]
    training = unsupervised["training"]
    print("Unsupervised Meta-World hidden-regime model")
    print(f"  unlabeled contexts per regime: {training['contexts_per_regime']}")
    print(f"  probe trials per context/probe: {training['probe_trials_per_probe']}")
    print(f"  action trials per context: {training['action_trials_per_context']}")
    print(f"  cluster purity: {diagnostics['cluster_purity']:.4f}")
    print(f"  probe likelihood MAE: {diagnostics['probe_likelihood_mae']:.4f}")
    print(f"  local section MAE: {diagnostics['section_mae']:.4f}")
    print()


def print_stream_summary(stream: dict[str, Any]) -> None:
    diagnostics = stream["fit_diagnostics"]
    training = stream["training"]
    counts = training["dataset"]["fingerprints_per_true_regime"]
    print("Stream unsupervised Meta-World hidden-regime model")
    print(f"  prior-sampled contexts: {training['total_contexts']}")
    print(f"  stream counts: {counts}")
    print(f"  probe trials per context/probe: {training['probe_trials_per_probe']}")
    print(f"  action trials per context: {training['action_trials_per_context']}")
    print(f"  cluster purity: {diagnostics['cluster_purity']:.4f}")
    print(f"  probe likelihood MAE: {diagnostics['probe_likelihood_mae']:.4f}")
    print(f"  local section MAE: {diagnostics['section_mae']:.4f}")
    print()


def print_raw_record_summary(raw_record: dict[str, Any]) -> None:
    diagnostics = raw_record["fit_diagnostics"]
    training = raw_record["training"]
    raw_counts = training["raw_dataset"]["contexts_per_true_regime"]
    event_counts = training["raw_dataset"]["records_per_event_type"]
    derived_counts = training["derived_dataset"]["fingerprints_per_true_regime"]
    print("Raw-record Meta-World hidden-regime model")
    print(f"  prior-sampled contexts: {training['total_contexts']}")
    print(f"  raw context counts: {raw_counts}")
    print(f"  derived fingerprint counts: {derived_counts}")
    print(f"  raw event counts: {event_counts}")
    print(f"  probe trials per context/probe: {training['probe_trials_per_probe']}")
    print(f"  action trials per context: {training['action_trials_per_context']}")
    print(f"  cluster purity: {diagnostics['cluster_purity']:.4f}")
    print(f"  probe likelihood MAE: {diagnostics['probe_likelihood_mae']:.4f}")
    print(f"  local section MAE: {diagnostics['section_mae']:.4f}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())

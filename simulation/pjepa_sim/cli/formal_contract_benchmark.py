"""Run the finite contract benchmark for verification adapters."""

from __future__ import annotations

from pjepa_sim.formal.contracts import run_formal_contract_benchmark, write_formal_contract_outputs


def main() -> None:
    results = run_formal_contract_benchmark()
    json_path, md_path = write_formal_contract_outputs(results)
    print("Formal contract benchmark")
    for agent, count in results["summary"]["passed_by_agent"].items():
        print(f"  {agent:>20s}  passed_contracts={count}/{len(results['suites'])}")
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()


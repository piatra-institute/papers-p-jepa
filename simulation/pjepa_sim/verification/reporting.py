"""Shared helpers for executable claim verifiers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_margin_report(
    results_path: Path,
    verify_path: Path,
    checks: Mapping[str, float],
    *,
    pass_on_zero: bool = False,
) -> int:
    """Write and print a pass/fail report from numeric claim margins.

    A positive margin means the claim passed. Some identity-preservation checks
    have zero as the exact expected value, so callers can opt into accepting
    zero margins as passing.
    """
    report = {
        "source": str(results_path),
        "checks": {
            name: {
                "margin": float(margin),
                "passed": bool(margin >= 0.0 if pass_on_zero else margin > 0.0),
            }
            for name, margin in checks.items()
        },
    }
    verify_path.parent.mkdir(parents=True, exist_ok=True)
    with verify_path.open("w") as f:
        json.dump(report, f, indent=2)

    failed = False
    for name, check in report["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status}  {name}: {check['margin']:.6f}")
        failed = failed or not check["passed"]
    print()
    print(f"Wrote: {verify_path}")
    return 1 if failed else 0


def strategy_aggregates(results: dict, *, nested: bool) -> dict[str, dict]:
    strategies = results["evaluation"]["strategies"] if nested else results["strategies"]
    return {
        name: result["aggregate"]
        for name, result in strategies.items()
    }

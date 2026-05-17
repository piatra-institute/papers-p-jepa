"""Run all local executable claim checks and write an audit summary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass

from pjepa_sim.verification.audit import LOCAL_VERIFIERS, VerifierSpec, write_claims_summary


@dataclass(frozen=True)
class RunResult:
    spec: VerifierSpec
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all local P-JEPA verifier modules.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Run remaining verifiers after a failure. By default the command stops on first failure.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not run verifiers; rebuild the claims summary from existing verifier JSON files.",
    )
    args = parser.parse_args()

    results: list[RunResult] = []
    if not args.summary_only:
        for spec in LOCAL_VERIFIERS:
            print(f"RUN   {spec.label}", flush=True)
            result = _run(spec)
            results.append(result)
            if result.passed:
                print(f"PASS  {spec.label}", flush=True)
            else:
                print(f"FAIL  {spec.label}", flush=True)
                _print_failure(result)
                if not args.keep_going:
                    sys.exit(result.returncode)

    json_path, md_path = write_claims_summary()
    print()
    print(f"Wrote: {json_path}", flush=True)
    print(f"Wrote: {md_path}", flush=True)

    failed = [result for result in results if not result.passed]
    if failed:
        sys.exit(1)


def _run(spec: VerifierSpec) -> RunResult:
    completed = subprocess.run(
        [sys.executable, "-m", spec.module],
        check=False,
        capture_output=True,
        text=True,
    )
    return RunResult(
        spec=spec,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _print_failure(result: RunResult) -> None:
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)


if __name__ == "__main__":
    main()

"""Hypothesis-driven ablation experiments for P-JEPA.

These runners test claims from docs/CRITIQUE.md against the existing code.
Each experiment has a preregistered pass/fail criterion documented in
docs/DELIVERY_PLAN.md and the plan file at
~/.claude/plans/ok-make-a-plan-declarative-scone.md. Results land in
docs/HYPOTHESIS_RESULTS.md.

The runners are pure additions; they do not modify any existing module
or CLI behaviour.
"""

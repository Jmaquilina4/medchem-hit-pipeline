"""Evaluation harness: the gated robustness report that produces the published metrics.

Scaffold-grouped cross-validation, a chronological split against era-split labels, a y-scrambled null,
class-balance and applicability-domain breakdowns, leakage measurement, and the configured pass/fail
gates. Runs on every invocation rather than as a one-off analysis.
"""

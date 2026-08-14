"""Virtual library screening (VLS) — the open-by-default tiered funnel (ADR 0005).

Tier 0 (``library``) prepares a purchasable deck; Tier 1 (``screen``) is the ligand
pre-filter that routes each compound by QSAR potency + direct-Δ selectivity + the
empirically-derived applicability domain. Structure tiers (docking → gnina → Boltz-2
→ FEP) are GPU work that drops in behind the same reason-coded manifest.
"""

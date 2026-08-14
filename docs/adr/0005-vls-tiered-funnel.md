# ADR 0005 — Virtual library screening: an open-by-default tiered funnel

**Status:** Accepted (design) · 2026-07 · *back-half; this ADR + the `vls:` config block are the spec, implementation follows the front-half close-out.*

> **Scope in this release.** This is a **design** record. Virtual screening ran on JAK1 only and is
> disabled in the BRD4 config; no docking, co-folding or free-energy campaign is part of the frozen
> results. Figures quoted below that come from that work are **not reproducible from this
> repository**, because its artifacts are not published here.


> ## Status of each tier, as shipped
>
> This ADR records a design that was partly built, partly retired and partly proposed. Read it with that
> in mind rather than as a description of the finished product:
>
> | element | status |
> |---|---|
> | Tier 0 / Tier 1 (physchem prefilter, applicability-domain stratification) | **implemented and run** — JAK1 only |
> | Docking tier | implemented; **no campaign is included in the frozen results** |
> | Co-folding tier | adapter defined; **explicit stub**, needs external GPU execution |
> | Free-energy tier | **retired** before use |
> | Open-vs-licensed engine head-to-head | **proposed, never run** |
>
> The frozen results depend on none of it: they cover data acquisition through gated evaluation. BRD4
> screening is disabled in its config.

## Context
The fast-follower thesis needs a way to turn a validated target + its known chemical matter into a **purchasable, synthesizable shortlist**. The v1 pipeline had no VLS. The front-half QSAR + direct-Δ selectivity models are the natural cheap first filter. VLS must stay reproducible on **open tools** (a stranger clones and runs); commercial tools (Schrödinger) are documented **optional per-tier swaps**, never dependencies.

Two facts shape the design:
1. **ATP-site conservation makes selectivity nearly invisible to docking.** JAK-family selectivity must come from the ligand-based direct-Δ model (Tier 1) and free energy (Tier 5) — never a docking score.
2. **The QSAR and a co-folding affinity head (Boltz-2) are both ChEMBL/BindingDB-trained** — they are one *data-correlated* opinion, not two independent votes. Only physics (FEP/ABFE) is genuinely data-orthogonal.

## Decision
A **6-tier funnel**, open-source on the default path; every tier emits a reason-coded drop manifest (no silent truncation):

| Tier | Purpose | Open default | Licensed-optional swap |
|---|---|---|---|
| **0** Library + prep | assemble + 2D-filter a purchasable deck | ZINC22 / Enamine REAL subsets; RDKit + Dimorphite-DL; PAINS + lead-like + hinge-pharmacophore | REAL Space (~94.5B) via infiniSee; physical kinase plates |
| **1** Ligand pre-filter | cheap 2D potency **+ selectivity** triage (the existing front half) | RF/XGB QSAR + direct-Δ selectivity + conformal AD gating | — |
| **2** Docking triage | ultra-large coarse 3D rank | Uni-Dock / Vina-GPU (Vinardo) | Glide HTVS/SP |
| **3** CNN rescore | sharpen true-/false-positive separation | gnina + smina rank-consensus | Glide XP |
| **4** Co-fold oracle | binder/non-binder **classifier** (NOT potency) | Boltz-2 (Boltzina on Tier-3 poses) | — (AF3/Chai gated; reference only) |
| **5** Physics tip | data-orthogonal free-energy anchor for the final dozens–hundreds | OpenFE / OpenMM RBFE; gmx_MMPBSA pre-rank | Prime MM-GBSA, FEP+, IFD-MD |

**Gate 0 — retrospective qualification before *any* novel hit is trusted:** qualify on a LIT-PCBA-style JAK1 active/inactive set (not DUD-E alone — property-matched decoys are learnable artifacts), **leave-class-out** split (hold out whole scaffold families), and the real bar = **rank JAK1-selective compounds above JAK2/3/TYK2 actives** (recovering pan-JAK binders is meaningless). Metrics panel (EF1/EF5/BEDROC-α20/PR-AUC/ROC-AUC) with bootstrap CIs vs a random baseline; validate **each engine** on the same fixed control set. Expect debiased numbers far below random-split/DUD-E numbers — **publish that drop; it's the honest signal.**

**Terminal gate + language discipline:** nothing is a "hit"/"inhibitor" until dose-response IC50 + orthogonal biophysics (SPR/TSA) + a selectivity counter-screen confirm it, with the **full batch hit-rate (incl. failures)** reported. Pre-assay outputs are **"prioritized candidates,"** never "inhibitors."

## Consequences
- Reproducible on open tools end-to-end; Schrödinger is a per-tier swap that buys integration/validation, **not** a categorical accuracy edge — and it breaks clone-and-run, so it's off the default path.
- Selectivity is carried by Tier 1 (ligand) + Tier 5 (physics) **by design** — never asked of docking.
- **QSAR + Boltz-2 agreement is logged but counted as ONE opinion**; the informative signals are the *disagreements* (QSAR↔Boltz-2, and Boltz-2↔FEP).
- Docking yields **enrichment, not affinity** (score↔affinity r≈0.4–0.6); the applicability domain bites hardest on the most novel hits — consistent with the [temporal-split finding](../RESULTS.md) (the QSAR is for near-domain analogs; novelty leans on structure + physics).
- Each tier is independently enable-able (composable/optional), so a project can run any prefix of the funnel.

## Config
See the `vls:` block in [`configs/jak1.yaml`](../../configs/jak1.yaml) (a new target = edit the receptor/library values, not code). It is `enabled: false` until built. The full annotated schema (Tier 0 prep, per-tier engines + gates, Gate-0 validation, prospective honesty gate) is captured there.

## Design refinements (2026-07-24)
- **This is the prospective fast-follower arm.** v1 used the ML models to rank/select selective hits from
  *existing* matter; VLS extends that *forward* — screen a large **purchasable** library to surface new
  buyable candidates. The temporal-AD finding motivates the funnel: the QSAR is strong for near-domain
  analogs (Tier 1) and weak on novelty, so structure (docking / Boltz-2 / FEP) carries the rest.
- **Library:** open default = **ZINC22** (free; aggregates Enamine REAL + WuXi + Mcule) and/or an
  **Enamine REAL kinase-focused lead-like subset**. The full REAL Space (~94.5B) is the *scale-navigation*
  story (FTrees / infiniSee / V-SYNTHES), not the exhaustively-docked deck. **Pin the snapshot.**
- **Compound size = LEAD-LIKE (MW ~300–460), NOT fragments.** Fast-following a drug-sized lead
  (upadacitinib ~380 Da) means analogs / scaffold-hops of drug-sized matter, and the QSAR is trained on
  drug-sized JAK compounds — fragments (MW <300, "rule of 3") would be out-of-domain (the AD term rejects
  them). Fragment-based screening is a different, novel-target strategy — not this.
- **Selectivity is NOT a docking signal** (ATP-site conservation) — it comes from Tier-1 ligand ML +
  Tier-5 physics. Docking yields enrichment, not affinity or selectivity.
- **Schrödinger = optional, license-gated plugin** behind the same interface (e.g. `structure/schrodinger.py`):
  a licensed adopter runs either Glide/MM-GBSA/IFD-MD/FEP+ **or** open Vina/gnina/Boltz-2/OpenFE. Open stays
  the default + CI path; the Schrödinger module is untested in the open repo (needs a license) and is never a
  dependency.
- **Open-vs-licensed docking head-to-head (proposed):** retrospective enrichment (EF1 / EF5 / BEDROC)
  on the *same* JAK1 actives/decoys set, comparing an open engine against a licensed one, to
  *data-justify* the open-default choice rather than asserting it. Not run; the licensed engine is
  outside this pipeline by design.
- **Scope impact:** this replaces the earlier licensed back end with the VLS funnel, the proposed
  head-to-head, and the constrained-generation reward-hacking arm.

## Tier-2 selection retired — a recorded reversal (2026-07-24)

**Superseded:** the Tier-1 `docking_budget: 50000`, the "far-novel (NN < 0.35) is never auto-docked" rule,
the `screening_cap: 100000` random sample, and the two-arm Tier-0.5 *selector*
(similarity-to-actives exploit + MaxMin explore). This is logged rather than silently edited, because the
reasoning is the interesting part.

**Why it was wrong.** All three rules rested on the premise *docking is the expensive step*. It isn't:
Uni-Dock fast mode puts a ~10⁶-compound library at roughly **one GPU-day** (~12–25 GPU-hours), so a 9×
deck cut saves ~20 GPU-hours while costing expected best-score and expected scaffold count — docking-score
vs library-size scaling is **log-linear with no measured saturation from 10⁵ to >10⁹** (Lyu, Irwin &
Shoichet 2023), and same-target prospective comparisons improved with size (AmpC 99M→1.7B: 11.4%→22.4%
hit rate). Three further findings, each of which independently breaks a rule we had:

1. **The random cap had exactly zero enrichment.** It captured 36 of the in-domain compounds available and
   5,298 near-domain — ≈ its own 10.7% sampling rate on *every* axis. So "the QSAR tier is idle" was
   partly self-inflicted: a coin flip had discarded ~90% of the only compounds the model could speak to.
2. **MaxMin is outlier selection dressed as sampling** — it hugs the periphery, and maximizing distance
   from the library centroid also maximizes distance from the *training set*, driving the deck further
   out-of-domain and making the QSAR **more** idle. It also buys nothing: for small subsets of a large
   diverse set, random ≈ MaxMin on diversity (Landrum, ChEMBL-35, 2025). Retired as a selector; where a
   representative diverse subset is genuinely needed, use **LeaderPicker sphere exclusion** (representative,
   not peripheral, and it scales) — and per Lyu et al. 2019, **cluster the docking output, never the input**
   (docking a single cluster representative measurably worsened top-ranked scores).
3. **Similarity-to-actives answers a different question than we were asking it.** It predicts where the
   ChEMBL-trained QSAR's ±0.955 conformal interval is *valid* (epistemic, about the model) and what is
   commercially interesting — it does **not** predict docking hit rate (prospective campaign hits sit at
   Tc < 0.6 to known actives, near random). Legitimate as a label, invalid as a gate.

**Decision.** Tier 1 **selects nothing**. It annotates every prepared compound and emits
applicability-domain **strata as reported labels** (S1 in-domain ≥ 0.50 · S2 near-domain 0.35–0.50 ·
S3 far-novel < 0.35). The whole Tier-0-prepared library is the docking queue — *"we screened the entire
sha256-pinned library"* needs no seed and satisfies the no-silent-truncation rule by construction. Deck
size becomes an **output** of Tier 0, never a configured constant.

**Budgets move to where compute is actually scarce**, and are pre-registered per stratum
(`vls.budget_allocation`), with the unstratified arm kept in the majority so the campaign does not become
a similarity search wearing a docking costume, and a **control arm ≥ 15%** so per-arm distributions stay
statistically separable:
- Tier 3 gnina (rescore-only; an orthogonal *artifact* filter, **not** a potency ranker) — 50k poses.
- Tier 4 Boltz-2 — 2,000, sampled across log-spaced score-rank bands + ≤3/scaffold rather than a pure
  top-N (measured to cut artifact content from ~100% to 25–50%).
- Tier 5 OpenFE — **`top_n` cut 200 → 40**. At ~6–12 GPU-hours *per ligand*, 200 was 50–100 GPU-days on
  one card. This was a real, unnoticed compute bug in the previous config.

**Three similarity terms, separated** (the previous single `nn_tanimoto` conflated two questions):
`sim_to_train` = applicability domain (defines strata) · `sim_to_actives` = priority/interest ·
`sim_to_known_reference` = **known-reference similarity ceiling**.

**The fast-follower trap, now explicit.** Patentability turns on 2D structure, so maximizing similarity to
upadacitinib-class actives maximizes the probability of landing inside the originator's
composition-of-matter claims — the obvious objective is anti-correlated with the commercial one. Marketed
references (upadacitinib, tofacitinib, baricitinib, filgotinib, abrocitinib; SMILES from ChEMBL, approved)
are flagged at `sim_to_known_reference ≥ 0.65` for review and **never silently dropped**. **Triage only — Tanimoto
has no legal standing**: Markush claims are defined by substitution patterns, so a compound at Tc 0.4 can
sit inside a claim while one at 0.7 sits outside.

**Known-unverified / carried risk.**
- The 12–25 GPU-hour figure is **borrowed** (a KRAS G12D campaign, someone else's box, exhaustiveness and
  prep) and could be 2–5× optimistic; protomer/tautomer expansion multiplies the job count. **Measure a
  5,000-compound pilot before quoting this budget as a result.** If the pilot is ≫ slower, fall back to
  **active learning** (seed with the complete near-domain enumeration + cluster-stratified far-novel, then
  acquire on predicted score), *not* to a random cap — AL converts the cap into a measurable recall number.
- The AD-supported arm is **correlated with the Tier-1 QSAR by construction** (prioritize by similarity to
  the training set, then score with a model trained on it = one opinion twice — the same failure mode this
  ADR already flags for QSAR + Boltz-2). Per-stratum enrichment claims must be made on Tier-2/3 output, the
  first genuinely independent signal, with the control stratum carried through unmodified.
- `sim_to_actives` inherits **ChEMBL publication bias** — the potent actives are what medicinal chemists
  chose to publish around a few successful JAK scaffolds, so proximity to them reproduces the field's
  historical preferences and under-ranks whatever JAK1 tolerates that nobody published.
- Hinge-motif SMARTS are **not yet trustworthy** (one plausible-looking hand-written pattern matched 0 of
  the actives). Validate any motif by reporting **recall on our own actives** alongside library retention,
  and never use one as a hard filter — a 2-point motif discards ~54% of known actives and the
  "privileged core" filter returns precisely the originator chemotype, destroying recall *and* baking in
  the IP we are trying to escape. Note also that an aromatic-N + adjacent-NH prior is type-I ATP-site
  specific and will systematically miss type-II/allosteric/covalent binders.
- **The deepest risk: the correct answer may be "wrong library,"** and this design could make the campaign
  look rigorous while still screening the wrong thing. Mitigation: run the in-stock library explicitly as a
  **published negative-result baseline** (in the campaign this ADR designs, max similarity-to-actives
  0.676 across 788,410 library compounds — a measurement from that campaign's artifacts, which are
  **not** part of this release; see the scope note at the top of this ADR), and in parallel move Tier 0 to the on-demand space (Enamine REAL / ZINC22 on-demand
  hinge-restricted tranches, or a vendor kinase catalogue ~4.7M dry-stock). V-SYNTHES-style hierarchical
  zoom does **not** apply to the current flat in-stock catalogue (45.8% singleton scaffolds — nothing to
  zoom into) but becomes non-optional the moment Tier 0 moves on-demand, where 29–95B products cannot be
  enumerated at all.


---

## SUPERSEDED IN PART (2026-08-04) — Tier 5 retired

**Tier 5 (alchemical free energy / OpenFE) is retired.** The tiered funnel above stands; its terminal
computational step does not. The pipeline now ends at a filtered, provenance-annotated candidate list
that hands off to experiment.

Reasoning, in short: adding a further layer does not reduce the dominant uncertainty. The figures below
come from the JAK1 virtual-screening and co-folding work this ADR designs, whose artifacts are **not
published in this release** — so they are recorded here as the basis for the decision and are **not
independently checkable from this repository**: QSAR MAE 1.28 log units in its operating band, the
co-folding model's average precision 0.0248 on its own benchmark, its two heads correlating at
rho = -0.14, and its affinity-value ensemble self-correlating at 0.34. Free energy stacked on that yields a *precise* number about a
compound whose identity was selected by noisy methods.

RBFE would also be operating outside its designed scope — it ranks close analogs within a validated
series whose binding mode is known crystallographically, whereas ours would be unsynthesised compounds,
a receptor we modelled ourselves (activation loop filled in where the crystal was disordered), and a
predicted pose.

**Accepted cost:** no computed selectivity number. Nothing else in the stack can supply one, so
selectivity is reported as unresolved with the experiment that would resolve it named. That is preferred
to a computed value on unmade compounds against a conserved ATP site.

Full reasoning is recorded in the working notes, which are not published with this repository.

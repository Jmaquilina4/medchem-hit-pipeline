# Pitfalls

Failure modes hit while building this pipeline, kept because **every one of them was silent** — the
tool reported success, produced plausible output, and exited zero. They are written from measurement,
not from documentation, and each is paired with the guard that now catches it.

If you are using any of these tools, this is the part of the repo most likely to save you time.

---

## Boltz-2 has three silent failure modes

All three are in the shipped 2.2.1 behaviour, all three keep going, all three exit `0`.

### 1. A mismatched MSA is replaced with a dummy, not an error

`featurizerv2.construct_paired_msa` compares the MSA's query row to the chain residues. On **any**
length or residue mismatch it prints `Warning: MSA does not match input sequence, creating dummy.`
and continues with **zero evolutionary information**. A one-residue difference silently converts a
whole campaign into single-sequence mode, and every prediction still "succeeds".

**Guard:** assert the MSA query row is byte-identical to the receptor sequence *before* fanning out,
and fail on the warning string. See `_assert_msa_matches`.

### 2. Per-record GPU OOM is swallowed

`boltz2.predict_step` catches `RuntimeError` containing `"out of memory"`, prints a warning, returns
`{"exception": True}`, and the exit code stays `0` (upstream #167). The record simply vanishes. The
affinity pass then dies with `FileNotFoundError` on the missing `pre_affinity_<id>.npz` (#390, #415) —
or, if it survives, you silently lose an arbitrary subset of your library.

**Guard:** count output JSONs against input count. Never treat exit status as a health signal.

### 3. Malformed inputs are skipped, and the run looks healthy

A YAML error produces `Failed to process <file>. Skipping.` — then Boltz loads both checkpoints,
reports `GPU available: True (cuda), used: True`, initialises bfloat16 AMP, and runs the Trainer over
**zero records**, looking entirely normal. We hit this with an eight-space indent where six was
required, which made `msa:` a continuation of the `sequence` scalar instead of a sibling key.

**Guard:** the YAML writer parses its own output back and asserts the sequence, SMILES, `msa` key and
affinity binder all survived; the shard counts `"Failed to process"` occurrences and refuses to
continue if any input was dropped.

### Two more Boltz-2 notes worth having

* **`--subsample_msa`'s help text is wrong.** It says "Default is True"; `is_flag=True` makes it
  `False` (#628). Enabling it runs an *unseeded* `torch.randperm` per forward pass, giving every
  ligand a different receptor MSA sample — variance injected directly into the quantity you are
  ranking. Bound MSA cost with `--max_msa_seqs` instead, and set `--seed`.
* **The affinity pass has its own budget.** `recycling_steps` is hard-coded to 5, and steps/samples
  come from `--sampling_steps_affinity` (default 200) and `--diffusion_samples_affinity` (default 5).
  Structure-pass flags do **not** reduce it. A "minimal" warm-up that only sets `--sampling_steps`
  runs the expensive pass at full cost — which is exactly how we burned 62 minutes believing we had
  configured a cheap run.
* **`ligand_iptm` is not a quality signal.** The affinity module was trained only on assays whose mean
  ipTM exceeded 0.75, so high ipTM is a *precondition of the training distribution*. "None below 0.7"
  is the expected outcome, not reassurance.

---

## Co-folding from a crystal structure: the gap trap

Boltz-2 and similar models fold from a **sequence**, so a PDB-derived receptor must be converted. The
obvious conversion is wrong in a way that never announces itself.

Our receptor (PDB 3EYG) has **280 observed residues spanning 865–1154 with three gaps totalling ten
unresolved residues**. Concatenating the observed residues — the natural thing to do — yields a 280-mer
with ten amino acids deleted across three artificial junctions: a protein that does not exist. The model
folds it happily and returns confident affinities.

**Guard:** splice the continuous sequence from UniProt, then *validate rather than assume* — every one
of the 280 crystal residues must match UniProt at identical numbering (ours did, 0 mismatches, which is
what confirms the numbering convention and makes the splice sound).

**And know what you filled in.** Our gap included **Y1034/Y1035**, the activation-loop tyrosines —
disordered in the crystal precisely because they are mobile. The co-folded receptor is therefore *not*
the docking receptor, which is a modelling choice to disclose rather than an implementation detail.

---

## Never scrape structured data out of terminal output without a checksum

We exported a 2,000-compound deck by returning it as JSON through a CLI and parsing the printed value.
**342 of 2,000 SMILES were corrupted** — bracketed aromatic atoms (`[nH]`) deleted — and the majority
parsed as **valid but different molecules**:

| | |
|---|---|
| true | `CN1CCN(c2[nH]c3cccc4c3c2-c2ccccc2C4=O)CC1` |
| corrupted | `CN1CCN(c2c3cccc4c3c2-c2ccccc2C4=O)CC1` — parses fine, wrong molecule |

The ten that failed loudly were the lucky ones. Removing an indole NH gives something RDKit accepts
without complaint, so downstream everything looks normal.

**Guard:** transport as base64 of the compressed bytes plus a SHA256, and verify. base64 contains no
quotes, backslashes or whitespace, so it survives any terminal or log mangling.

---

## `.gitignore` does not strip inline comments

```
build/     # kept out of the repo
```

Git treats the **entire string including the comment** as the pattern, so it matches nothing. Two files
inside that directory declared themselves "gitignored, local-only" in their own headers. They were not.

This is a *semantics* error, so no secret scanner or grep can catch it — a leak scan would report the
files and you would dismiss the finding because you believed they were excluded.

**Guard:** put every comment on its own line, and verify with `git check-ignore -v <path>` rather than
reading the file and assuming.

---

## Remote container builders may strip quotes from build commands

A build step written as:

```python
.with_commands(['python3 -c "import os; assert ..."'])
```

arrived at the shell as `python3 -c import` — a `SyntaxError`, after the build had already spent seven
minutes downloading 5.78 GiB. Quote characters did not survive the layer that hands commands to `sh`.

**Guard:** use quote-free shell in build commands (`test $(stat -c %s f) -eq N`), and move anything
needing real quoting into a task that runs *after* the build, where a real interpreter exists.

---

## Long-running subprocesses must stream, not buffer

`subprocess.run(capture_output=True)` buffers until exit. A run that hung for 62 minutes produced
**nothing** in the logs — only the runtime's own banner. There was no way to distinguish "queued behind
a slow server" from "spinning in a retry loop", which is why it had to be killed rather than diagnosed.

**Guard:** stream each line to stdout as it arrives while also accumulating it. Any subprocess that can
run for minutes needs this, or a stall is indistinguishable from progress.

---

## Do not put a network call inside a GPU reservation

The same 62-minute stall was an HTTP round-trip to a public MSA server, executed inside a task holding
an L40S. The card sat idle polling a web API, and the vendored client loops
`while status in ["UNKNOWN","RATELIMIT"]: sleep(5+rand); resubmit` with **no bound**, so a server-side
throttle is indistinguishable from slowness.

**Guard:** the MSA fetch is a **CPU-only** task with a bounded timeout. It completes in ~1 second. Only
the prediction ever needed the GPU. Separate steps by *what resource they need*, not by what feels tidy.

---

## Assorted tool-specific traps

| tool | trap |
|---|---|
| `pdb2pqr` | `--pH` is a **propka stability** option. Protonation state is set by `--with-ph`; using the wrong one silently protonates at default 7.0. |
| `meeko` | `--keep-chain` shifted PQR column positions, producing a parse error downstream. Dropped. |
| AutoDock Vina | `--write_maps` requires `--force_even_voxels` **and** a ligand, and keys maps by Vina's own internal atom types (`N_D`, `C_P`, …), hard-erroring on any type absent from the cache. Grid caching was **rejected** on measurement, not principle: the cache cost more to build and validate than the re-computation it replaced. |
| RDKit `ReplaceSidechains` | Prefer it over hand-rolled bond surgery for scaffold extraction. Our graph-walking version broke aromatic rings and produced invalid SMILES; the purpose-built call handled spiro fusion correctly first time. |
| Random forests | Cannot extrapolate — predictions are bounded by training values. A *high* prediction in a sparse region is more likely trees averaging over unrelated leaves than genuine signal, so a model can be most confident exactly where it is least competent. |

---

## The meta-lesson

Every GPU failure catalogued above was diagnosable on CPU: wrong subprocess flags, a YAML parse error,
a shell-quoting problem. **Validate every artifact a GPU will consume before requesting the GPU.** The
cost asymmetry is large and entirely one-directional.

And the reason all of the above is written down: the tools' *default posture on failure is to continue*.
Guards that count outputs, re-parse inputs, and assert invariants are not defensive programming here —
they are the only thing standing between a clean-looking run and a fabricated result.

---

## A leak scanner that exempts its own file cannot see its own leaks

This one is worth stating because the failure is structural, not careless.

The scanner in this repo keeps organisation-specific terms in an untracked overlay, so that publishing
the scanner does not publish the strings it detects. Sound idea. But it also carried a path-level
allowlist exempting **its own file**, on the reasoning that its pattern definitions would otherwise
match themselves.

That exemption made it blind to **13 genuine blocking matches inside its own source**, including an
organisation email domain embedded in a live pattern, an employment reference in the module docstring,
and a vendor name used as a pattern identifier. The scanner reported the tree CLEAN while carrying those
strings itself.

**Fix:** exempt only the pattern-definition block, delimited by sentinels, and scan the rest of the file
like any other. Write patterns using character classes (`targ[e]t audien[c]e`) so the file does not
literally contain the phrases it blocks.

## A too-narrow pattern is worse than no pattern

The same scanner missed three occurrences of an organisation name because the regex required the
compound form (`acmecorp`, `acme ai`) while the documents used the **bare word** on its own, followed by
an ordinary noun. It reported CLEAN, which is more dangerous than reporting nothing, because a clean
report gets taken as evidence.

**Fix:** include the bare token, and treat any clean result from a hand-written pattern list as
provisional until something independent — a second scanner, or a reviewer — has looked.

**General lesson for both:** a tool that reports on itself needs an external check. Every leak this
scanner missed was found by a human reading the files.

## "Optional" verified at the graph level, crashing at the call level

A stage was made an *optional* dependency: the DAG composes without it, dependent stages run, and the
consumer drops the missing component from its scoring spec instead of scoring it as zero. Tests asserted
all of that — and all of it was true.

The consumer still crashed. It passed `selectivity_predict=None` into a scorer that called the predictor
unconditionally, before any spec-driven branch, so the reward function raised
`TypeError: 'NoneType' object is not callable` the first time the stage body actually executed. The
composition tests could not see it: they assert on the **plan**, and a plan never runs anything.

The type checker had been reporting it the whole time — `Argument of type "Any | None" cannot be assigned
to parameter "selectivity_predict" of type "Predict"` — filed under 26 errors dismissed as
"mostly third-party stub gaps".

**Fixes, all three needed:**
- Type the optional thing as optional (`Predict | None`) so the checker follows it to every call site.
- Test at the layer that **executes**, not only the layer that plans.
- Make the degenerate case loud: if a spec still requests the component while no model was supplied,
  raise. Scoring it as a constant would invent a measurement and silently shift every reward.

## Resolving an alias in the library but not at the entry point

Stages were re-registered under a generic pipeline name with an alias mapping the old target-named slug
to the new one. `runner.plan("jak1")` resolved it, a test asserted both slugs produced identical graphs,
and the test passed.

Every documented command still failed. The CLI validated the raw `--pipeline` value against the registry
*before* calling `plan`, so `-p jak1` exited with a usage error while the internal API it wrapped worked
perfectly. The README had already been updated to promise the alias worked.

**Fix:** resolve at the boundary, and write the regression test through the **entry point users actually
type** (`CliRunner`), not the internal function. A compatibility shim that isn't tested where the user
enters is not a shim.

**Shared shape with the entry above:** both were verified one layer away from where they break. Testing
the plan instead of the run, and the library instead of the CLI, is the same mistake twice.

## A local gate that is not literally CI's commands is a different gate

pyright reported zero errors locally and one error in CI: `Import "meeko" could not be resolved`.
Nothing was wrong with the code. `meeko` lives in an optional `docking` extra that was installed on
the development machine and not in CI, so the type checker was analysing a different program in each
place. The red build was the *first* signal, arriving after the push.

The tempting fix is to suppress the import. That hides the divergence instead of removing it, and the
same class of gap remains for every other optional dependency — including three ligand-prep tests
that were skipping in CI *and* locally, so they ran nowhere at all.

**Fix:** CI installs the extra (it is pure Python — no CUDA, no licence), and `scripts/gate.sh` runs
CI's commands, in CI's order, with CI's dependency set. `tests/test_ci_parity.py` parses the workflow
and fails if the two ever drift apart, because keeping two lists in sync by intention does not work.

## A config knob nothing reads, and a config spec that could never run

An AST audit of every config key the package actually reads turned up two failure modes in one file.

**Keys read by nothing.** `model.potency.algorithms`, `model.potency.calibration`, and
`generative.scoring.aggregation` all looked like knobs. Random forest and XGBoost both always train,
conformal calibration always runs, and both aggregations are always computed because their difference
*is* the reward-hacking evidence. Setting `algorithms: [random_forest]` would have trained XGBoost
anyway. They were deleted, not modeled: a knob that does nothing is worse than no knob.

**A spec that could not execute.** Three scoring components declared `transform: sigmoid` with no
`center`, and `sigmoid` has no default centre. That spec would raise the moment it was scored. It
survived because the stage used a hard-coded default spec and ignored the configured one — so the
config was decorative, and both the error and the fact that config was being ignored stayed hidden.
Honouring config immediately produced `sigmoid() missing 1 required positional argument: 'center'`.

**Fix:** validate specs against the real transform signatures by introspection (`inspect.signature`),
never by duplicating a requirements list — so adding a required parameter to a transform
automatically invalidates every config that omits it. Unknown parameters are rejected too, because
`centre` would otherwise be silently dropped.

**The pattern across all of these:** configuration and behaviour drift apart silently, and tests that
assert on plans, graphs, or library functions cannot see it. Only executing the thing can.

## The co-folding guards, and why they belong in the package rather than a launch script

Guarding the three silent failures above inside an out-of-package execution layer, hardcoded to a
single target, is the wrong home: the guards are the reusable part, and a second target would have to
re-derive them from a wasted GPU run.

`medchem.structure.cofold` now holds them, working from a sequence, a ligand and an optional MSA:

* `write_boltz_yaml` **parses back what it wrote** before returning. Indentation errors are caught
  per-file by the framework and reported as a skipped input, after which it runs over zero records
  while appearing busy. Validating on CPU at write time costs microseconds; discovering it on a card
  cost one run at 0/4 records.
* SMILES is single-quoted and the round trip **compares the parsed value to the input**, so a nitrile
  truncated at a `#` comment fails at write time instead of becoming a different molecule.
* `assert_msa_matches` requires the `key,sequence` CSV and compares the query sequence **exactly**,
  reporting the first differing index. Any mismatch would otherwise be replaced by a dummy alignment
  with nothing in the output saying so — and the affinity head was trained with real alignments.
* A relative MSA path is rejected outright: it resolves against the process working directory, not the
  input file.
* The protein sequence is **derived from the fetched structure**, preferring SEQRES over observed
  atoms. The flagship target used a committed FASTA with a hand-annotated header; observed-atom
  fallback silently shortens the chain at every disordered loop, so which source was used is recorded.

**The general lesson:** a guard written in a launch script protects one campaign. The same guard in the
package protects every target, and its cost is a unit test instead of a GPU reservation.

## A scanner that reports CLEAN over zero files

The binary-metadata scanner discovered candidates with `git ls-files`. Run against a **sanitized
export** — which by design has no `.git` — that command returns nothing, and the scanner printed:

    CLEAN — no identifying metadata in 0 binary file(s).

The export is precisely the tree that most needs checking, and it was the one tree the scanner could not
see. The count was in the output the whole time; nobody reads a count when the verdict says CLEAN.

**Fix, two parts:** fall back to a filesystem walk when `git ls-files` fails, and **refuse to report
clean on zero candidates** — a pass over nothing is not a pass. The same repository had already been
bitten by a leak scanner that exempted its own file and by one whose pattern was too narrow to match.
Three variants of one failure: *the tool reported success by looking at nothing, or at the wrong thing.*

**Related trap on the same path.** The export directory is not a git repository, so `git config
user.email` there resolves to the **global** config rather than the repository-local override. Running
`git init` in an export and committing would embed whatever the global identity happens to be, which
is not necessarily the one the repository intends to publish under.
The leak scanner catches it because it checks the identity that the *next* commit would use, not the one
history happens to contain.

## Documenting a leak by quoting it reintroduces the leak

Writing up a history finding, the paragraph naming the two exposed terms was itself flagged by the
scanner — correctly. The document described the problem by reproducing it, which put the terms back into
the tracked tree at the moment of explaining that they should not be there.

**Fix:** describe the class, not the string. "An authentication acronym and a term naming the
orchestration platform's managed infrastructure layer" carries the same information for a reader and
matches no pattern. This is the same technique the scanner already uses on its own definitions, where
patterns are written with character classes so the file does not contain the phrases it blocks.

**Why it is worth a section:** the instinct when documenting a leak is to be specific, and specificity is
usually a virtue here. This is the one place it inverts — and the scanner is what catches it, which is an
argument for running the gate before every commit rather than trusting a careful author.

## A rank statistic on parallel predictions is not reproducible to float precision

`RandomForestRegressor(n_jobs=-1).predict()` is **not bitwise reproducible**, even for one fitted model in
one process. Scikit-learn accumulates each tree's contribution into a shared array from a thread pool, so
the summation order follows whatever order the threads finish in. Measured on 400 trees: repeated
`predict()` calls on the same model differ by up to **1.3 × 10⁻¹⁵**. With `n_jobs=1` they are exactly
identical.

That difference is far too small to matter for a continuous metric — R², RMSE, a confidence bound. The
trap is that **ROC-AUC and PR-AUC are not continuous in the predictions**; they are functions of the
predicted *ordering*. A 10⁻¹⁵ shift that happens to straddle two near-tied predictions swaps their ranks,
and the AUC jumps by one discrete quantum — which for a few thousand pairs is on the order of **10⁻⁷**,
eight orders of magnitude larger than the perturbation that caused it.

The failure mode this produces is a reproduction check that passes on most artifacts and fails on a few,
with deltas that look far too large to be rounding and therefore look like a real discrepancy. It is easy
to misread as a broken pipeline. The diagnostic is the *pattern*: continuous metrics agreeing to ~10⁻¹⁶
while only rank statistics disagree, and only for the datasets that contain near-ties.

**Guards:**

* State reproduction tolerances **per metric family**, not one number for the whole repository. A single
  tolerance is either too loose to be meaningful for the continuous metrics or too tight to be true for
  the rank ones.
* Decide deliberately between determinism and stability of published numbers. Setting `n_jobs=1` buys
  bitwise reproducibility, but if results are already frozen and published it also means re-running them
  to change values that alter no conclusion.
* Check that the *decisions* built on the metric are unaffected. A support verdict that comes from a
  threshold comparison, or from a confidence interval spanning zero, has margins many orders of magnitude
  wider than a rank quantum, and is genuinely exact where the underlying metric is not.

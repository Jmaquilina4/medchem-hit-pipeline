#!/usr/bin/env bash
# THE checks. One executable path, used by CI and by a developer, so "it passes locally" and "it passes
# in CI" cannot mean different things.
#
# CI runs `bash scripts/gate.sh --ci`. There is no second list of commands in the workflow to drift from
# this one, which is what the earlier arrangement had: the workflow enumerated the steps, this file
# enumerated them again, and a textual parity test compared the two lists. That test could only ever
# check that the strings matched -- it could not catch this file failing to RUN, which is exactly what
# happened when a step here imported a development-only file that the sanitized export does not ship.
#
# Two modes, and the difference is only ever what a runner cannot do:
#
#   (default, developer)  additionally requires a repository-local noreply git identity, because a commit
#                         made here would inherit whatever is configured. Strictly a superset of --ci.
#   --ci                  a runner creates no commits, so that finding is unreachable there.
#
# SELF-CONTAINED: every command below runs in the sanitized export as well as in the development tree.
# Steps that only make sense in one of them detect which they are in and say so, rather than failing.
#
# Usage:  bash scripts/gate.sh          # developer
#         bash scripts/gate.sh --ci     # exactly what the runner executes
set -euo pipefail

cd "$(dirname "$0")/.."

CI_MODE=0
for arg in "$@"; do
  case "$arg" in
    --ci) CI_MODE=1 ;;
    *) printf 'unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

step "Sync environment (pinned via uv.lock)"
uv sync --extra science --extra dev --extra docking --frozen

step "Lint"
uv run ruff check .

step "Lint the shipped scripts"
# In the EXPORT, pyproject's extend-exclude is narrowed and `ruff check .` above already covers
# scripts/ -- the gate's own checkers must not be unchecked code where they are the only tooling a
# reader has. In the DEVELOPMENT tree scripts/ is excluded, because legacy one-offs live there and one
# does not even parse, so a lint error in a SHIPPED script would be invisible here and fail only after
# an export was built. That happened once: an import left unused by a refactor.
#
# So this step lints the shipped set explicitly WHEN it can determine it, and says so when it cannot.
# The list comes from the export allowlist if that file is present, and otherwise this step is a no-op
# because `ruff check .` has already done the work.
uv run python - <<'PYEOF'
import pathlib, subprocess, sys

exporter = pathlib.Path("scripts/export_public.py")
if not exporter.is_file():
    print("  this tree lints scripts/ in the sweep above (no separate allowlist needed here)")
    sys.exit(0)

import importlib.util as iu
spec = iu.spec_from_file_location("_ep", exporter)
assert spec is not None and spec.loader is not None
ep = iu.module_from_spec(spec)
spec.loader.exec_module(ep)
files = [f"scripts/{n}" for n in ep.KEEP_SCRIPTS
         if n.endswith(".py") and pathlib.Path("scripts", n).is_file()]
print(f"  linting {len(files)} shipped script(s) with the export's rules")
sys.exit(subprocess.run(["uv", "run", "ruff", "check", "--no-force-exclude", *files]).returncode)
PYEOF

step "Types"
uv run pyright

step "Tests"
uv run python -m pytest tests/ -q

step "Provenance identity records this tree's scientific-source digest"
uv run python scripts/scientific_source_digest.py --check-identity

step "Frozen snapshots verify against the published checksums"
uv run python scripts/publish_snapshots.py --check

step "Published provenance records are present and internally consistent"
uv run python scripts/publish_provenance.py --check

step "Frozen figures depict the provenance records they ship beside"
uv run python scripts/make_frozen_figures.py --check

step "Temporal-overlap records agree with the published reports"
uv run python scripts/derive_temporal_overlap.py --check

step "Docs match the frozen manifests"
uv run python scripts/verify_docs_against_manifests.py

step "Reproduction report matches the published provenance"
uv run python scripts/check_reproduction_report.py

step "Release readiness (history, snapshot content, links, documented paths)"
# Reports SKIPPED, distinctly from CLEAN, on a tree that is not a sanitized export: a development tree
# legitimately contains in its history the material these checks look for.
uv run python scripts/check_release_ready.py --export .

if [ "$CI_MODE" -eq 1 ]; then
  step "Leak scan (--ci: a runner creates no commits, so an absent local identity is not a finding)"
  uv run python scripts/check_no_leaks.py --ci
else
  step "Leak scan (developer mode: also requires a repository-local noreply identity)"
  uv run python scripts/check_no_leaks.py
  step "Leak scan (--ci: the same scan a runner performs)"
  uv run python scripts/check_no_leaks.py --ci
fi

step "Binary metadata scan"
uv run python scripts/check_binary_metadata.py

# MANDATORY when the exporter is present: build a sanitized export, clone it, and run THIS gate inside
# it. Nothing cheaper catches a gate that passes here and fails there, because that failure mode is a
# missing FILE rather than a mismatched string -- and it happened.
#
# The export does not ship export_public.py, so the inner gate's copy of the test skips and the nesting
# terminates one level deep. Skipped entirely when this tree IS the export.
if [ -f scripts/export_public.py ]; then
  step "Exported gate runs in a sanitized fresh clone (integration)"
  uv run python -m pytest tests/test_ci_parity.py -m integration -q
else
  step "Exported gate integration (skipped: this tree is the export)"
  printf '  this tree ships no exporter, so there is nothing to export and re-run\n'
fi

printf '\n\033[1;32mAll gates passed.\033[0m\n'
if [ "$CI_MODE" -eq 0 ]; then
  printf 'Note: the leak scan loads scripts/leak_patterns_local.toml when present. It is untracked,\n'
  printf 'so CI runs the generic patterns only -- a CLEAN result in CI is weaker than one here.\n'
fi

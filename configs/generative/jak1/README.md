# REINVENT configs — substituting `${REPO}`

Every TOML in this directory contains `${REPO}` placeholders. They are **not** shell variables and
REINVENT does not expand them: the committed form is deliberately path-independent so that no config
encodes one machine's directory layout. Substitute the repository root before use.

```bash
# from the repository root
REPO="$(pwd)"
for f in configs/generative/jak1/*.toml; do
  sed "s|\${REPO}|${REPO}|g" "$f" > "/tmp/$(basename "$f")"
done
```

Then point REINVENT at the substituted copies in `/tmp`, not at the committed originals.

## What the placeholders resolve to

| placeholder use | resolves to |
|---|---|
| `params.executable = "${REPO}/.venv/bin/python"` | the interpreter created by `uv sync` |
| `params.args = "${REPO}/scripts/reinvent_oracle.py --config ${REPO}/configs/jak1.yaml"` | the scoring bridge, and the target config whose models it loads |

`--config` is **mandatory** on the bridge. It was made required after an earlier version defaulted to a
target, which meant a config could silently score against the wrong model;
`tests/test_config.py::test_every_reinvent_config_passes_the_required_oracle_config` asserts that every
TOML here supplies it, and that test is written to fail rather than pass when it finds no configs.

## Which config does what

| file | arm |
|---|---|
| `sampling_r1.toml` | sample-and-filter baseline — no reinforcement learning |
| `scoring.toml` | the scoring function used by the baseline |
| `armB_rl.toml` | reinforcement-learning arm at matched sampling budget |
| `armB_v2_rl.toml` | the same arm after a reward-specification correction |
| `scoring_v2.toml` | corrected scoring function: `FractionCSP3` as a band rather than one-sided |
| `scaffolds.smi` | the carried-forward scaffolds used as generation seeds |

`scoring_v2.toml` exists because the first reward specification was one-sided on `FractionCSP3`, which
rewarded saturation without bound. The uncorrected version is kept so the correction is visible rather
than tidied away.

## Note on outputs

The generated compound sets from these runs are **not** published in this repository. They were scored
by a potency model trained on the pooled assay cohort, which is not the model the frozen headline
results use, so their scores are not comparable to anything in
[`docs/RESULTS.md`](../../../docs/RESULTS.md). The configs and the bridge ship so the arm is
reproducible; the stale outputs do not.

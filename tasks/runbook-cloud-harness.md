# Runbook: first runs of `cloud/harness` on the 2080 Ti and the 5080

Every command here runs from the repo root. Go in order and stop at the first step that does not
pass; keep its printed output and the JSON file it names. Nothing on this branch has run on
torch yet (see [overnight-cloud-2026-11-26](overnight-cloud-2026-11-26.md)), so step 1 is the
first time most of it executes.

The 2080 Ti is free, so it goes first. The 5080's data build and probes need no GPU and can run
at the same time; its GPU steps wait until BitDraft is done.

## 0. Get the branch

The branch is in the project folder as `chained-transformer/cloud-harness/cloud-harness.bundle`.
On the 2080, with the bundle downloaded:

```sh
git fetch /path/to/cloud-harness.bundle cloud/harness:cloud/harness
git switch cloud/harness
```

On the 5080, take it from the 2080 over Tailscale:

```sh
git fetch 2080 cloud/harness:cloud/harness
git switch cloud/harness
```

`pyproject.toml` and `uv.lock` are unchanged, so no `uv sync` is needed. Pushing the branch to
`origin` is a separate decision.

## 1. 2080 Ti: check

```sh
uv run --locked --no-sync python -m tools.check
```

Pass: exit 0 with no `failed:` lines. Most likely to fail, in order:

- `types`: mypy ran here without torch installed, so every torch type was `Any`. Errors that
  only real torch stubs show will surface now.
- `tests`: the tolerance tests (`test_chunked_adamw8bit_tracks_fp32_adamw`, the bf16 MoE
  closeness tests in `test_surgery.py`) and the bit-exact chunking tests in `test_optim.py`.

## 2. 2080 Ti: memory at 4k and 8k (must fit)

```sh
uv run --locked --no-sync python -m tools.memory_check --device cuda --base Qwen/Qwen3-0.6B --seq-len 4096 8192 --budget-gib 10.5 --out artifacts/memory-check/2080-fixed.json
```

Pass: exit 0 and ten lines (five arms, two lengths), each ending in `fits`, none with
`OUT OF MEMORY` or `attention fell back to math`. The estimates were, in GiB:

| arm | 4k | 8k |
| --- | ---: | ---: |
| sft | 4.2 | 5.1 |
| full | 7.2 | 8.2 |
| lora | 3.7 | 4.6 |
| moe | 7.4 | 8.5 |
| separate | 4.2 | 5.1 |

Each line prints `ratio` (measured over predicted). Between about 0.7 and 1.3 means the
estimate is usable as is; record the ratios either way, they are what calibrates
`chained_transformer.memory`.

## 3. 2080 Ti: memory at 16k (informative)

```sh
uv run --locked --no-sync python -m tools.memory_check --device cuda --base Qwen/Qwen3-0.6B --seq-len 16384 --budget-gib 10.5 --out artifacts/memory-check/2080-fixed-16k.json
```

Expected: `sft`, `lora` and `separate` fit; `full` and `moe` (estimated 10.0 and 10.8 GiB
before the CUDA context) are likely `OVER` or out of memory, so exit 1 here is expected and is
why those two arms train on the 5080. To see whether frozen embeddings bring them in
(estimated 9.4 and 10.2):

```sh
uv run --locked --no-sync python -m tools.memory_check --device cuda --base Qwen/Qwen3-0.6B --arms full moe --seq-len 16384 --freeze-embeddings --budget-gib 10.5 --out artifacts/memory-check/2080-fixed-16k-frozen.json
```

## 4. 2080 Ti: the old harness, for the before and after (optional)

```sh
uv run --locked --no-sync python -m tools.memory_check --device cuda --base Qwen/Qwen3-0.6B --harness current-turing --seq-len 4096 --budget-gib 10.5 --out artifacts/memory-check/2080-current.json
```

Expected: every arm `OUT OF MEMORY`, most in `optimizer`, as in last night's smoke, and exit 1.
If the old harness fits now, last night's failure had a cause the overnight write-up missed.

## 5. 2080 Ti: MoE in bf16

```sh
uv run --locked --no-sync python -m tools.gpu_equivalence --variants full moe --out artifacts/equivalence/2080-moe-bf16.json
```

Each printed line is: variant, dtype, identical tokens, first divergence, max logit difference,
first mode switches. Pass: every `moe` line reads `moe torch.bfloat16 True None`, like the
`full` lines, with a max logit difference about as small as `full`'s. Before the fix, `moe`
diverged with differences of 0.4 to 0.66.

## 6. 5080, no GPU needed: data build

```sh
uv run --locked --no-sync python -m chained_transformer.data --policy equal-domains-16k
```

It first prints `reading N row groups, X GiB compressed` and stops adding row groups at 6 GiB
(`--max-download-gb`). Pass: exit 0, and `artifacts/data/ot3-mix/` holds the dataset,
`stats.json` and `sample.jsonl`. In `stats.json`, `selected.by_domain` should list code, math
and science, and `policies` shows, for each candidate policy, its loss-token share per mode and
its rows per domain. Read a few rows of `sample.jsonl` to see the mode boundaries look right.

For the 2080 Ti's later training runs, the 8k version
(`--policy equal-domains-8k --out artifacts/data/ot3-mix-8k`), copied over with `rsync`.

## 7. 5080, no GPU needed: mode probes

```sh
OMP_NUM_THREADS=8 uv run --locked --no-sync python -m tools.mode_probe --data artifacts/data/ot3-mix
```

Add `--device cuda` instead when a GPU is free. Pass: exit 0 and
`artifacts/probes/mode-probe.json` with, for `position_mode` and `segment_label`, a balanced
accuracy per layer (layer 0 is the embedding) and a `majority_baseline`. What to read from it:
the first layer where accuracy is well above the baseline is where the modes are already
linearly separable, which is where the design note's option 3 (mode-specific ends, shared
looped middle) would start specialising.

## 8. 5080, when the GPU is free: GPU smoke

Check `nvidia-smi` and `pueue status` first; never stop another project's job.

```sh
uv run --locked --no-sync python -m tools.gpu_smoke --data artifacts/data/ot3-mix
```

Without step 6's data, leave `--data` off to use the old `ot3-slice-4k`. Pass: every step
prints `ok`, `smoke.json` under `artifacts/smoke/<timestamp>/` has every `passed` entry true,
and exit 0. In its `train_summaries`, each arm should show `attention_math_fallbacks: 0`;
`train-full-16k`'s `peak_reserved_gib` is the 16k memory answer for the 5080's 16 GB.

## Afterwards

Record memory, step times and the probe table in [first-experiment](first-experiment.md), copy
the compact JSON into `results/`, and settle open decisions 1 (training data) and 2 (sequence
length) there.

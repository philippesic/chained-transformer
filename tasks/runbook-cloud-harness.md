# Runbook: first runs of `cloud/harness` on the 2080 Ti and the 5080

Every command runs from the repo root and can be pasted as is. Go in order and stop at the
first step that does not pass; keep its printed output and the JSON file it names. Nothing on
this branch has run on torch yet (see [overnight-cloud-2026-11-26](overnight-cloud-2026-11-26.md)),
so step 1 is the first time most of it executes.

The 2080 Ti goes first (steps 1 to 6). On the 5080, the data build and probes (steps 7 and 8)
need no GPU and can run at the same time; its GPU steps (9 and 10) wait until BitDraft is done.

## Sharing the GPUs

The 2080 Ti also runs ShaderLOD's jobs, in pueue's `gpu` group. This project's group there is
`ct`, one job at a time, so the 2080 Ti's long steps (2 to 6) are written as `pueue add -g ct`
commands. Queue them one at a time or all at once; read them in order with
`pueue status -g ct` and `pueue log --full <id>`. "Exit 0" below means the task shows `Success`,
and the printed lines are in its log. To run a step in the terminal instead, drop its
`pueue add -g ct --label <name> --` prefix (and, in step 5, the outer quotes). If `pueue add`
says the group doesn't exist, create it with `pueue group add ct && pueue parallel 1 -g ct`.

Steps 1 and 1b take a minute or two; run them directly once `pueue status` shows no job running
in `gpu`. The memory check and the GPU smoke refuse to measure while another process holds the
GPU (exit 2); requeue the step when ShaderLOD's job ends, and never stop that job.

[AGENTS.md](../AGENTS.md) asks for jobwatch on jobs longer than a few minutes. I don't know its
command, so wrap these the way you usually do.

## Expected memory

Peak allocated GiB from `chained_transformer.memory`, without the CUDA context. The
coefficients are not calibrated yet, so a measurement anywhere in the range (0.7 to 1.3 times
the estimate) is expected. Passing is decided by the budget, not by the range.

| arm | 4k | 8k | 16k |
| --- | --- | --- | --- |
| sft | 4.2 (2.9 to 5.5) | 5.1 (3.6 to 6.6) | 6.9 (4.8 to 9.0) |
| full | 7.2 (5.0 to 9.4) | 8.2 (5.7 to 10.7) | 10.0 (7.0 to 13.0) |
| lora | 3.7 (2.6 to 4.8) | 4.6 (3.2 to 6.0) | 6.5 (4.5 to 8.5) |
| moe | 7.4 (5.2 to 9.6) | 8.5 (5.9 to 11.1) | 10.8 (7.6 to 14.0) |
| separate | 4.2 (2.9 to 5.5) | 5.1 (3.6 to 6.6) | 6.9 (4.8 to 9.0) |

`--freeze-embeddings` takes about 0.6 off every cell. Budgets: 10.5 GiB on the 2080 Ti (11 GB
card), 15 GiB on the 5080 (16 GB card).

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

Pass: exit 0 with no `failed:` lines.

If it fails: mypy ran here without torch installed, so every torch type was `Any`, and errors
only real torch stubs show will surface now. A `types` failure alone doesn't block the GPU
steps; see the tests on their own with:

```sh
uv run --locked --no-sync pytest -q -p no:cacheprovider
```

A failing test in `test_optim.py`, `test_attention.py` or the MoE tests in `test_surgery.py`
does block them, because those are the pieces the GPU steps rely on. Keep the output.

## 1b. 2080 Ti: GPU unit tests

```sh
uv run --locked --no-sync pytest -m gpu -q -p no:cacheprovider
```

These run on tiny random models, so they need no download. On CUDA they check fused attention
in bf16 and fp32 with no fallback to math, the 8-bit optimizer against its CPU result,
unbiased bf16 rounding, that an optimizer step's temporaries stay within one chunk, that MoE
with checkpointing frees its recomputed activations, and three steps of each arm.

Pass: every test passes.

If it fails: unlike the other steps, go on to step 2 anyway, because its numbers show what the
failure costs. Keep the output. A failing attention test, or `attention_math_fallbacks` above 0
in the arm tests, means the fused kernel refused these shapes; the assertion prints its error.
A failing MoE memory test means the `moe` rows of steps 2, 3 and 6 will read high.

## 2. 2080 Ti: memory at 4k and 8k (must fit)

```sh
pueue add -g ct --label 2-memory-4k-8k -- uv run --locked --no-sync python -m tools.memory_check --device cuda --base Qwen/Qwen3-0.6B --seq-len 4096 8192 --budget-gib 10.5 --out artifacts/memory-check/2080-fixed.json
```

Pass: exit 0 and ten lines (five arms, two lengths), each ending in `fits`, none with
`OUT OF MEMORY` or `attention fell back to math`. Note each line's `ratio` (measured over
estimated); they calibrate `chained_transformer.memory`.

If it fails:

- Exit 2 and `GPU busy`: another process holds the GPU, most likely a ShaderLOD job; requeue
  when it ends.
- `OUT OF MEMORY in optimizer`: the chunked optimizer is not the one running, or its chunk is
  too big. Check the arm's `setup` in the JSON says `"optimizer": "chunked"`.
- `OUT OF MEMORY in forward` or `backward`, or `attention fell back to math`: attention is
  likely on the math kernel. The warning in the output gives the shape and the reason. Retry
  the arm with `--loss-chunk 128 --freeze-embeddings` to see how far off it is.
- `OVER` on `full` or `moe` at 8k only: those two have the least margin; record it and keep them
  at 4k on this card.

## 3. 2080 Ti: memory at 16k (informative)

```sh
pueue add -g ct --label 3-memory-16k -- uv run --locked --no-sync python -m tools.memory_check --device cuda --base Qwen/Qwen3-0.6B --seq-len 16384 --budget-gib 10.5 --out artifacts/memory-check/2080-fixed-16k.json
```

Expected: `sft`, `lora` and `separate` fit; `full` and `moe` are likely `OVER` or out of
memory, so exit 1 here is expected and is why those two train on the 5080. To see whether frozen
embeddings bring them in:

```sh
pueue add -g ct --label 3-memory-16k-frozen -- uv run --locked --no-sync python -m tools.memory_check --device cuda --base Qwen/Qwen3-0.6B --arms full moe --seq-len 16384 --freeze-embeddings --budget-gib 10.5 --out artifacts/memory-check/2080-fixed-16k-frozen.json
```

## 4. 2080 Ti: the old harness, for the before and after (optional)

```sh
pueue add -g ct --label 4-old-harness -- uv run --locked --no-sync python -m tools.memory_check --device cuda --base Qwen/Qwen3-0.6B --harness current-turing --seq-len 4096 --budget-gib 10.5 --out artifacts/memory-check/2080-current.json
```

Expected: every arm `OUT OF MEMORY`, most in `optimizer`, as in last night's smoke, and exit 1.
If the old harness fits now, last night's failure had a cause the overnight write-up missed.

## 5. 2080 Ti: 4k training smoke on real rows

Twenty steps of each arm on the existing slice:

```sh
pueue add -g ct --label 5-smoke-4k -- 'for arm in sft full lora moe; do uv run --locked --no-sync python -m chained_transformer.train --arm $arm --data artifacts/data/ot3-slice-4k --out artifacts/runs/2080-4k-$arm --steps 20 --seq-len 4096 --complete-only --device cuda || exit 1; done; uv run --locked --no-sync python -m chained_transformer.train --arm separate --mode answer --data artifacts/data/ot3-slice-4k --out artifacts/runs/2080-4k-separate --steps 20 --seq-len 4096 --complete-only --device cuda'
```

The task stops at the first arm that fails.

Pass: each run ends with `training complete`, every `step=` line shows a finite loss, and each
`artifacts/runs/2080-4k-<arm>/summary.json` has `attention_math_fallbacks: 0` and a
`peak_alloc_gib` no higher than step 2's measurement for that arm. Rows longer than 4,096 tokens
are cut here (`truncated_rows` in `train.jsonl`), which is fine for a smoke.

If it fails: an out-of-memory here but not in step 2 means real rows behave differently from
the memory check's synthetic ones; keep the run's `train.jsonl`. A `nan` loss on one arm only
points at that arm's surgery; on every arm, at the optimizer.

## 6. 2080 Ti: MoE in bf16

```sh
pueue add -g ct --label 6-moe-bf16 -- uv run --locked --no-sync python -m tools.gpu_equivalence --variants full moe --out artifacts/equivalence/2080-moe-bf16.json
```

Each printed line is: variant, dtype, identical tokens, first divergence, max logit difference,
first mode switches. Pass: every `moe` line reads `moe torch.bfloat16 True None`, like the
`full` lines, with a max logit difference about as small as `full`'s. Before the fix, `moe`
diverged with differences of 0.4 to 0.66.

If it fails: if `full` diverges too, the drift is in generation, not in the MoE combine. If only
`moe` does, note how large the logit difference is now against 0.4 to 0.66 before.

## 7. 5080, no GPU needed: data build

Steps 7 and 8 run for a while on the CPU. Wrap them the way you usually do on the 5080, but not
in its pueue `gpu` group, which would hold them behind BitDraft.

```sh
uv run --locked --no-sync python -m chained_transformer.data --policy equal-domains-16k
```

It first prints `reading N row groups, X GiB compressed` and stops adding row groups at 6 GiB
(`--max-download-gb`). Pass: exit 0, and `artifacts/data/ot3-mix/` holds the dataset,
`stats.json` and `sample.jsonl`. In `stats.json`, `selected.by_domain` lists code, math and
science, and `policies` shows each candidate policy's loss-token share per mode and rows per
domain. Read a few rows of `sample.jsonl` to see the mode boundaries look right.
[docs/data.md](../docs/data.md) explains every field.

If it fails: an HTTP error from Hugging Face is most likely the network or the Hub, since the
existing slice came from the same public dataset; retry. If `N` is far below 400, the row groups
are bigger than expected; raise `--max-download-gb` or accept fewer rows.

For the 2080 Ti's later training runs, also build the 8k version and copy it over (put the
2080's repo path in place of `<repo>`):

```sh
uv run --locked --no-sync python -m chained_transformer.data --policy equal-domains-8k --out artifacts/data/ot3-mix-8k
rsync -a artifacts/data/ot3-mix-8k 2080:<repo>/artifacts/data/
```

## 8. 5080, no GPU needed: mode probes

```sh
OMP_NUM_THREADS=8 uv run --locked --no-sync python -m tools.mode_probe --data artifacts/data/ot3-mix
```

Add `--device cuda` instead when a GPU is free. Pass: exit 0 and
`artifacts/probes/mode-probe.json` with, for `position_mode` and `segment_label`, a balanced
accuracy per layer (layer 0 is the embedding) and a `majority_baseline`. The first layer well
above the baseline is where the modes are already linearly separable, which is where the design
note's option 3 (mode-specific ends, shared looped middle) would start specialising.

## 9. 5080, when the GPU is free: memory at 16k for `full` and `moe`

Only once the 5080 is free for this project's GPU work. Check `nvidia-smi` and `pueue status`
first and never stop another project's job; [development](../docs/development.md) says to queue
behind BitDraft in the 5080's `gpu` group, so put `pueue add -g gpu --label 9-memory-16k --` in
front of the command below, and likewise for step 10. The check refuses to start while another
process holds the GPU.

```sh
uv run --locked --no-sync python -m tools.memory_check --device cuda --base Qwen/Qwen3-0.6B --arms full moe --seq-len 16384 --budget-gib 15 --out artifacts/memory-check/5080-16k.json
```

Pass: exit 0, both lines end in `fits` (estimates 10.0 and 10.8 GiB). If either is `OVER`,
rerun with `--freeze-embeddings`; if that fits, 16k training on this card needs that flag.

## 10. 5080, when the GPU is free: GPU smoke

```sh
uv run --locked --no-sync pytest -m gpu -q -p no:cacheprovider
uv run --locked --no-sync python -m tools.gpu_smoke --data artifacts/data/ot3-mix
```

The first line repeats step 1b on this card, which takes different attention kernels; its
pass and fail read as there.

Without step 7's data, leave `--data` off to use the old `ot3-slice-4k`. Pass: every step
prints `ok`, `smoke.json` under `artifacts/smoke/<timestamp>/` has every `passed` entry true,
and exit 0. In its `train_summaries`, each arm shows `attention_math_fallbacks: 0`;
`train-full-16k`'s `peak_reserved_gib` is the 16k memory answer for the 5080.

## Afterwards

Record memory, step times and the probe table in [first-experiment](first-experiment.md), copy
the compact JSON into `results/`, and settle open decisions 1 (training data) and 2 (sequence
length) there.

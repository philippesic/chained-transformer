# Demo: one cache with mode swaps against separate models that re-prefill

The chained model serves every mode from one vLLM engine and one KV cache: a mode switch only
changes which MLP the next tokens use. The alternative is one model per mode, where every switch
moves the request to another engine that has to prefill the history again. This demo measures
both on the same prompts with `tools.bench_vllm`, and `tools.mode_swap_table` turns the result
into one table. Nothing here has run on a GPU yet; the expectations below come from reading the
code, not from a run.

## What each arm does

- `base`: plain Qwen3-0.6B, one engine. The reference for speed, prefill and tokens.
- `chained/select` and `chained/triton`: the chained export in one engine with the plugin.
  `select` runs every mode's MLP on every token and keeps one (three times the MLP compute, the
  safe path); `triton` runs each token through its own mode only.
- `separate`: three engines, one per mode, each with its own cache and a third of the memory.
  Generation runs in segments of 16 tokens so a switch is noticed; tokens generated after a
  switch inside a segment are thrown away, and the request moves to the new mode's engine, which
  prefills whatever of the history it has not cached. Prefix caching is on, the best case for
  this arm.

With identical modes (every mode a copy of the base MLP), every arm should produce the base
model's tokens, so the run checks correctness and cost together. With modes that differ,
re-prefilling also changes the answer: the new engine recomputes the earlier tokens with its
own MLP, which is not how the chained model was trained. The optional run in step 5 shows that.

## Before you start

- Get the branch, on the 2080 Ti host:
  ```sh
  cd /home/philip/chained-transformer
  git fetch /path/to/cloud-serving-review.bundle cloud/serving-review:cloud/serving-review
  git switch cloud/serving-review
  ```
- The vLLM virtualenv from step 2 of `MORNING.md` (`~/venvs/vllm`, vLLM 0.34.0). That install is
  still your call; nothing below works without it.
- `Qwen/Qwen3-0.6B` from the Hugging Face Hub, downloaded or cached on the host.
- Nothing else on the card: check `nvidia-smi` and `pueue status`. `tools.bench_vllm` does not
  check, and a shared card skews every number.
- Run everything from the repo root, so `$PWD` is the repo. Each prepared folder holds about
  6 GB of exports (an estimate for Qwen3-0.6B in 16-bit).
- Like `MORNING.md`, these use plain pueue and assume `ct` jobs land on the 2080 Ti; wrap them in
  jobwatch the way you usually do.

## On the 2080 Ti (pueue group `ct`)

The 2080 Ti has no bfloat16, so the exports and engines run in float16. `ct` runs one job at a
time in the order you add them, so you can add all of steps 1 to 4 at once.

### 1. Export the models

Exports the chained model and one plain model per mode, all with copies of the base MLP, plus
64 chat prompts with thinking on. This step runs on the CPU.

```sh
pueue add -g ct -- env PYTHONPATH=$PWD ~/venvs/vllm/bin/python -m tools.bench_vllm prepare --dir artifacts/demo-swap/same --model qwen --dtype float16 --num-prompts 64
```

### 2. Check float16 on this card

Qwen3 activations can overflow float16. This runs the base model in float32 and in float16 on
the same exported weights and compares the tokens.

```sh
pueue add -g ct -- env PYTHONPATH=$PWD VLLM_ENABLE_V1_MULTIPROCESSING=0 ~/venvs/vllm/bin/python -m tools.bench_vllm arm --dir artifacts/demo-swap/same --arm base --dtype float32 --max-tokens 256 --gpu-mem 0.8 --max-model-len 1024 --json-out artifacts/demo-swap/base-fp32.json
pueue add -g ct -- env PYTHONPATH=$PWD VLLM_ENABLE_V1_MULTIPROCESSING=0 ~/venvs/vllm/bin/python -m tools.bench_vllm arm --dir artifacts/demo-swap/same --arm base --dtype float16 --max-tokens 256 --gpu-mem 0.8 --max-model-len 1024 --json-out artifacts/demo-swap/base-fp16.json
pueue add -g ct -- uv run --locked --no-sync python -m tools.compare_outputs artifacts/demo-swap/base-fp16.json artifacts/demo-swap/base-fp32.json --window 32 --out artifacts/demo-swap/fp16-vs-fp32.json
```

The last job prints one line, for example `61 of 64 prompts agree on the first 32 tokens (12 on
every token); 0 and 0 outputs repeat one token from the start`. Greedy float16 and float32
drift apart over long outputs, so few whole-output matches is normal. Treat float16 as broken
if its outputs repeat one token where float32's don't, or if fewer than half the prompts agree
on the first 32 tokens (a rule of thumb, not a measured threshold). Then add `--dtype float32`
to the `all` commands below; the demo's point holds in float32, only slower.

### 3. Run the demo with identical modes

Runs `base`, `chained/select`, `chained/triton` and `separate` one after another, each in its
own process, up to 80% of the card for each. 1,024 new tokens per prompt, so thinking has room
to end and the answers have room for code, which is where the switches come from.

```sh
pueue add -g ct -- env PYTHONPATH=$PWD ~/venvs/vllm/bin/python -m tools.bench_vllm all --dir artifacts/demo-swap/same --max-tokens 1024 --gpu-mem 0.8 --max-model-len 2048 --out artifacts/demo-swap/2080-same.json
```

### 4. Make the table

```sh
pueue add -g ct -- uv run --locked --no-sync python -m tools.mode_swap_table artifacts/demo-swap/2080-same.json --out artifacts/demo-swap/2080-same.md
pueue add -g ct -- uv run --locked --no-sync python -m tools.compare_outputs artifacts/demo-swap/same/row_chained_triton.json artifacts/demo-swap/same/row_base_na.json --window 32 --out artifacts/demo-swap/2080-same-triton-vs-base.json
```

### 5. Optional: modes that differ

Adds noise to the copies of the base MLP for modes 1 and 2 (5% of each weight's mean size, a
starting guess), so routing changes the tokens. Expect `same as base` to drop for every arm
except `base`, and `separate` to disagree with `chained/select` after its first switch.

```sh
pueue add -g ct -- env PYTHONPATH=$PWD ~/venvs/vllm/bin/python -m tools.bench_vllm prepare --dir artifacts/demo-swap/perturbed --model qwen --dtype float16 --num-prompts 64 --perturb 0.05
pueue add -g ct -- env PYTHONPATH=$PWD ~/venvs/vllm/bin/python -m tools.bench_vllm all --dir artifacts/demo-swap/perturbed --max-tokens 1024 --gpu-mem 0.8 --max-model-len 2048 --out artifacts/demo-swap/2080-perturbed.json
pueue add -g ct -- uv run --locked --no-sync python -m tools.mode_swap_table artifacts/demo-swap/2080-perturbed.json --out artifacts/demo-swap/2080-perturbed.md
pueue add -g ct -- uv run --locked --no-sync python -m tools.compare_outputs artifacts/demo-swap/perturbed/row_chained_select.json artifacts/demo-swap/perturbed/row_separate_na.json --window 32 --out artifacts/demo-swap/2080-perturbed-chained-vs-separate.json
```

Once the pilot has trained a `full` or `lora` checkpoint, `prepare --checkpoint PATH` in place
of `--perturb 0.05` runs the same demo on real modes.

## On the 5080

5080 only, once BitDraft is done and Philip says so. The same demo in bfloat16, in its own
folder. No pueue group is set aside for the 5080, so queue these the way you run 5080 jobs.
bfloat16 has float32's range, so the float16 check is not needed.

```sh
env PYTHONPATH=$PWD ~/venvs/vllm/bin/python -m tools.bench_vllm prepare --dir artifacts/demo-swap/same-bf16 --model qwen --dtype bfloat16 --num-prompts 64
env PYTHONPATH=$PWD ~/venvs/vllm/bin/python -m tools.bench_vllm all --dir artifacts/demo-swap/same-bf16 --dtype bfloat16 --max-tokens 1024 --gpu-mem 0.8 --max-model-len 2048 --out artifacts/demo-swap/5080-same.json
uv run --locked --no-sync python -m tools.mode_swap_table artifacts/demo-swap/5080-same.json --out artifacts/demo-swap/5080-same.md
uv run --locked --no-sync python -m tools.compare_outputs artifacts/demo-swap/same-bf16/row_chained_triton.json artifacts/demo-swap/same-bf16/row_base_na.json --window 32 --out artifacts/demo-swap/5080-same-triton-vs-base.json
```

For modes that differ, repeat step 5 with `--dtype bfloat16` on both `bench_vllm` commands and
a folder of its own, such as `artifacts/demo-swap/perturbed-bf16`.

## Reading the table

| Column | Meaning |
|---|---|
| tok/s | new tokens per second over the timed generation |
| prefill tokens | prompt tokens the engines computed; tokens served from the prefix cache are not counted |
| extra prefill vs base | prefill tokens above `base` |
| prefill after a switch | prefill on each request's first call in the engine it moved to (`separate` only): the re-prefill cost of switching |
| switches | mode changes the tracker saw in the generated text (counted by `separate`) |
| wasted tokens | tokens generated after a switch inside a segment and thrown away |
| same as base | prompts whose whole output equals `base`'s |
| mixed steps | chained steps whose batch held more than one mode |

What a good run looks like, from the code rather than a run:

- `chained/select` and `chained/triton` have 0 extra prefill: a switch costs no cache work.
- `separate` has extra prefill. `prefill after a switch` is the part that comes from moving
  to another engine and grows with `switches`. The rest is the harness restarting each request
  every 16 tokens, which recomputes up to one cache block each time; a streaming setup with
  separate models would not pay that, so read the switch column as the cost of separate models.
- With identical modes, `same as base` should be 64 for `chained/select` (the serving checklist
  warns when it is not). `chained/triton` and `separate` compute in a different order, so a few
  outputs may drift late; the `triton-vs-base` comparison should show nearly all prompts
  agreeing on the first 32 tokens.
- `chained/triton` runs close to `base`, `chained/select` below it, and `separate` lowest. The
  chained arms copy each step's tokens to the host for the mode tracker, which stalls vLLM's
  async scheduling, so their speed understates the plugin.
- `mixed steps` above 0 shows batches really held several modes at once.

If `switches` is 0, thinking never ended within 1,024 tokens. Rerun step 3 with
`--max-tokens 2048 --max-model-len 3072`.

## What to paste back

- `artifacts/demo-swap/fp16-vs-fp32.json` or its one printed line.
- `artifacts/demo-swap/2080-same.md`, and `2080-perturbed.md` if you ran step 5.
- `pueue log --full <id>` for any job that failed; `pueue status -g ct` lists the ids.

# Development

## Environment

CPython 3.11.15 and uv 0.11.32 on Linux x86-64, matching the BitDraft checkout on this host.
`pyproject.toml` pins direct dependencies and `uv.lock` pins everything: torch 2.14.0 with
CUDA 13.0 wheels (PyPI default), transformers 5.24.1, datasets 5.2.0, torchao 0.18.0. The
`eval` extra adds math-verify and evalplus.

```sh
uv sync --locked --extra eval
git config core.hooksPath .githooks
```

To update a dependency, change its pin in `pyproject.toml`, run `uv lock`, then the check and
the GPU smoke. Model weights and datasets download into the Hugging Face cache
(`~/.cache/huggingface`); `python -m chained_transformer.data` writes slices under `artifacts/data/`.

## Checks

| Lane | Command | Covers | Time |
| --- | --- | --- | --- |
| Check (pre-commit hook) | `uv run --locked --no-sync python -m tools.check` | ruff lint and format, mypy, pytest on CPU | ~7 s |
| Real-model equivalence on CPU | `OMP_NUM_THREADS=4 uv run --locked --no-sync python -m tools.gpu_equivalence --device cpu --dtype float32 --max-new-tokens 160 --out artifacts/cpu-checks/equivalence.json` | Qwen3-0.6B: identical modes give the base model's greedy tokens across mode switches | ~3 min |
| Memory check (GPU) | `uv run --locked --no-sync python -m tools.memory_check --device cuda --base Qwen/Qwen3-0.6B --seq-len 4096 8192 16384 --budget-gib 10.5` | Every arm, 3 training steps each in its own process: peak allocated and reserved GiB per phase, measured against `chained_transformer.memory`'s estimate; exits 1 if any arm runs out of memory or exceeds the budget. Add `--harness current-turing` to reproduce the old setup | a few minutes per length (estimate) |
| Memory check (CPU) | `uv run --locked --no-sync python -m tools.memory_check --device cpu --layers 2 --seq-len 512 1024` | The same comparison on a random model with Qwen3-0.6B's widths, counting live tensor bytes per phase; calibrates the estimate's coefficients for the fixed harness. It cannot see inside tensor subclasses, so torchao's temporaries are invisible: measure `--harness current-turing` on the GPU | minutes (estimate) |
| GPU unit tests | `uv run --locked --no-sync pytest -m gpu -q -p no:cacheprovider` | `tests/test_gpu.py` on tiny random models: fused attention without math fallback, the 8-bit optimizer on CUDA, MoE checkpoint memory, a fresh MoE matching the base over 1,024 batched greedy tokens in bf16, three steps of each arm in bf16 and fp16, a captured decode graph matching the uncaptured step for each arm. Any CUDA GPU, no download | a minute or two (estimate) |
| Conversion check on CPU | `OMP_NUM_THREADS=4 uv run --locked --no-sync python -m tools.conversion_check --round-trip --out artifacts/cpu-checks/conversion.json` (add `--random-shape` to skip the download) | Fresh `full`/`lora`/`moe` conversions of Qwen3-0.6B against the base: prefill logits per routing pattern, cached decoding across switches, bitwise save/load | ~1 min |
| Real-model smoke | `uv run --locked --no-sync python -m tools.real_model_smoke` | Needs the Hub. Conversion check with real weights, fresh `full`/`lora`/`moe` checkpoints, scorer oracles (MATH500, HumanEval+), then greedy smoke evals of the base and each fresh checkpoint on 3 problems per benchmark; `summary.json` says whether each checkpoint's responses are identical to the base's | ~20 min on a GPU (estimate) |
| Eval CLI plumbing on CPU | `uv run --locked --no-sync python -m tools.tiny_checkpoint --out artifacts/tiny/full` then `python -m evals.run --model artifacts/tiny/full --benchmarks humanevalplus --limit 3 --greedy --max-new-tokens 64 --out artifacts/smoke/tiny` | Generation, mode tracking, scoring and the report on a random tiny model with a toy tokenizer; no Hub access | ~1 min |
| Upgrade demo on CPU | `uv run --locked --no-sync python -m tools.upgrade_demo --report results/upgrade-demo/report.json` | Single-mode code upgrade on a tiny synthetic task, two seeds, against plain fine-tunes; see [the result](../results/upgrade-demo/result.md) | ~15 min |
| Pilot (GPU) | `uv run --locked --no-sync python -m tools.pilot plan --name NAME --data DATA`, then `pueue add --group ct -- uv run --locked --no-sync python -m tools.pilot run --root artifacts/pilot/NAME` on the 2080 Ti | By default the verdict set: `sft`, `full`, `lora`, `moe` x two seeds, each trained, saved and evaluated greedily on a sample sized for a 2-point gap that grows if the arms disagree more than assumed (`size` shows it); `widen` adds `separate` and the full suite, resuming the evals, and `--eval-set full` plans everything at once; `status` and `compare` read the runs and apply the kill rule; `python -m tools.pilot_report --root artifacts/pilot/NAME` prints the paper tables and the verdict. Needs `cloud/harness` merged (the `on_model` hook in `train.main`) | hours (not run) |
| Eval power | `uv run --locked --no-sync python -m tools.eval_power` | Problems per benchmark the kill rule needs to see a 2-point gap at two seeds, for several disagreement rates | instant |
| Pilot budget | `pueue add -g ct -- uv run --locked --no-sync python -m tools.pilot_budget measure --device cuda --base Qwen/Qwen3-0.6B --seq-len 4096` on the 2080 Ti, then `python -m tools.pilot_budget estimate --root artifacts/pilot/NAME --measured 2080ti=artifacts/pilot-budget/measure.json` | A few training steps per arm, timed by phase, then the pilot's hours per arm on each GPU; without `--measured` it is a FLOP-model estimate with its assumptions printed. Refuses a busy GPU | ~5 min per GPU (estimate) |
| Decode bench (GPU) | `pueue add -g ct -- uv run --locked --no-sync python -m tools.decode_bench --out artifacts/bench/2080.json` on the 2080 Ti | Decode step time at 7x1024, 7x4096, 7x8192 and 16x2048 cached tokens for `--attention auto` and `lean`, each with `DynamicCache` and the preallocated cache; a fitted fixed cost and cost per 1,000 cached tokens; how often each choice picks the first one's token. Refuses a busy GPU. `--device cpu --shapes 6x1024 6x3072 --steps 8` is the CPU form | ~4 min on a GPU (estimate), ~10 min on CPU |
| Eval time from lengths | `uv run --locked --no-sync python -m tools.eval_sim --responses DIR/math500/responses.jsonl --problems 830 --fixed-ms 38 --ms-per-1000 2.7` | Replays the fixed-batch and the grouped refill over real response lengths under a step-time model (the decode bench's fit) and prints hours, rows per step and tokens per second | seconds |
| Loss against the base | `uv run --locked --no-sync python -m tools.loss_vs_base --run-dir artifacts/pilot/NAME/sft-s0 --steps 0 1 20 272-275 --calibration-below 2 --out artifacts/detect/sft-s0.jsonl` | Base-model loss in fp32 on CPU on exactly the rows each logged step of an `sft` run trained on, so the difference from the logged loss is the training effect with row difficulty cancelled | ~6 min per step on 4 threads at 8k |
| GPU smoke | `uv run --locked --no-sync python -m tools.gpu_smoke` | bf16 and fp32 equivalence on the GPU, a 50-step `full` run, 10-step runs of the other arms, one 16k-token memory probe | ~15 min (estimate) |

The check lane uses a tiny random Qwen3 and the cached Qwen3-0.6B tokenizer, so it needs the
tokenizer downloaded once. CPU results are not GPU or performance evidence.

`tools.gpu_smoke` refuses to start while another process holds the GPU and lists the
processes. The RTX 5080 is reserved for BitDraft and is not used until Philip says so. On the
RTX 2080 Ti this project queues in pueue's `ct` group (parallelism 1), as in
`pueue add -g ct -- uv run --locked --no-sync python -m tools.serving_gpu_checklist`; the `gpu`
group there belongs to ShaderLOD and is never used. Check `nvidia-smi` and `pueue status` before
GPU work and never stop another project's job.
Outputs go to `artifacts/smoke/<timestamp>/` (`smoke.json` plus one log per step).

## Two machines

Both checkouts track the public GitHub repository `origin`
(`git@github.com:philippesic/chained-transformer.git`). Commit and push on one machine, then pull
on the other:

```sh
git push
git pull --ff-only
```

Each checkout also names the other as a remote over the Tailscale SSH aliases (`2080` on the
5080, `5080` on the 2080), for pulling directly when GitHub is unreachable. `artifacts/` is not
in Git; copy what a run needs with `rsync`. The 2080 Ti has 11 GB and compute capability 7.5,
so its memory and speed numbers do not stand in for the 5080's.

## Conventions

- No code comments or docstrings; use names, types and tests. Rationale belongs in these docs
  or commit messages.
- Product code in `chained_transformer/` never imports `tools/` or `tests/`.
- Everything large stays in ignored `artifacts/`. Compact evidence that a result needs is
  copied into `results/<name>/`.
- Push to `origin` (public) at meaningful milestones. Ask before publishing
  elsewhere.

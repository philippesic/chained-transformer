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
| GPU unit tests | `uv run --locked --no-sync pytest -m gpu -q -p no:cacheprovider` | `tests/test_gpu.py` on tiny random models: fused attention without math fallback, the 8-bit optimizer on CUDA, MoE checkpoint memory, three steps of each arm. Any CUDA GPU, no download | a minute or two (estimate) |
| Conversion check on CPU | `OMP_NUM_THREADS=4 uv run --locked --no-sync python -m tools.conversion_check --round-trip --out artifacts/cpu-checks/conversion.json` (add `--random-shape` to skip the download) | Fresh `full`/`lora`/`moe` conversions of Qwen3-0.6B against the base: prefill logits per routing pattern, cached decoding across switches, bitwise save/load | ~1 min |
| Real-model smoke | `uv run --locked --no-sync python -m tools.real_model_smoke` | Needs the Hub. Conversion check with real weights, fresh `full`/`lora`/`moe` checkpoints, scorer oracles (MATH500, HumanEval+), then greedy smoke evals of the base and each fresh checkpoint on 3 problems per benchmark; `summary.json` says whether each checkpoint's responses are identical to the base's | ~20 min on a GPU (estimate) |
| Eval CLI plumbing on CPU | `uv run --locked --no-sync python -m tools.tiny_checkpoint --out artifacts/tiny/full` then `python -m evals.run --model artifacts/tiny/full --benchmarks humanevalplus --limit 3 --greedy --max-new-tokens 64 --out artifacts/smoke/tiny` | Generation, mode tracking, scoring and the report on a random tiny model with a toy tokenizer; no Hub access | ~1 min |
| Upgrade demo on CPU | `uv run --locked --no-sync python -m tools.upgrade_demo --report results/upgrade-demo/report.json` | Single-mode code upgrade on a tiny synthetic task, two seeds, against plain fine-tunes; see [the result](../results/upgrade-demo/result.md) | ~15 min |
| Pilot (GPU) | `uv run --locked --no-sync python -m tools.pilot plan --name NAME --data DATA`, then `pueue add --group gpu -- uv run --locked --no-sync python -m tools.pilot run --root artifacts/pilot/NAME` | Five arms x two seeds (`separate` is three fine-tunes per seed), each trained, saved as a checkpoint and evaluated greedily; `status` and `compare` read the runs and apply the kill rule; `python -m tools.pilot_report --root artifacts/pilot/NAME` prints the paper tables and the verdict. Needs `cloud/harness` merged (the `on_model` hook in `train.main`) | hours (not run) |
| Pilot budget | `uv run --locked --no-sync python -m tools.pilot_budget measure --device cuda --base Qwen/Qwen3-0.6B --seq-len 4096`, then `python -m tools.pilot_budget estimate --root artifacts/pilot/NAME --measured 5080=artifacts/pilot-budget/measure.json` | A few training steps per arm, timed by phase, then the pilot's hours per arm on each GPU; without `--measured` it is a FLOP-model estimate with its assumptions printed. Refuses a busy GPU | ~5 min per GPU (estimate) |
| GPU smoke | `uv run --locked --no-sync python -m tools.gpu_smoke` | bf16 and fp32 equivalence on the GPU, a 50-step `full` run, 10-step runs of the other arms, one 16k-token memory probe | ~15 min (estimate) |

The check lane uses a tiny random Qwen3 and the cached Qwen3-0.6B tokenizer, so it needs the
tokenizer downloaded once. CPU results are not GPU or performance evidence.

`tools.gpu_smoke` refuses to start while another process holds the GPU and lists the
processes. The RTX 5080 is shared with BitDraft: check `nvidia-smi` and `pueue status` before
GPU work, never stop another project's job, and queue behind it in pueue's `gpu` group instead.
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

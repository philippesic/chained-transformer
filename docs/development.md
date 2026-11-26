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
| GPU smoke | `uv run --locked --no-sync python -m tools.gpu_smoke` | bf16 and fp32 equivalence on the GPU, a 50-step `full` run, 10-step runs of the other arms, one 16k-token memory probe | ~15 min (estimate) |

The check lane uses a tiny random Qwen3 and the cached Qwen3-0.6B tokenizer, so it needs the
tokenizer downloaded once. CPU results are not GPU or performance evidence.

`tools.gpu_smoke` refuses to start while another process holds the GPU and lists the
processes. The RTX 5080 is shared with BitDraft: check `nvidia-smi` and `pueue status` before
GPU work, never stop another project's job, and queue behind it in pueue's `gpu` group instead.
Outputs go to `artifacts/smoke/<timestamp>/` (`smoke.json` plus one log per step).

## Two machines

The 5080 checkout is the main one; the 2080 Ti checkout tracks it (`main` follows `5080/main`).
Each checkout names the other as a remote over the Tailscale SSH aliases. Git refuses pushes
into a checked-out branch, so move commits by pulling on the receiving machine:

```sh
git pull --ff-only            # on the 2080: take the 5080's commits
git pull --ff-only 2080 main  # on the 5080: take the 2080's commits
```

Commit work on one machine before pulling on the other. `artifacts/` is not in Git; copy what
a run needs with `rsync`. The 2080 Ti (11 GB, compute capability 7.5) has no fast bf16, so its
memory and speed numbers do not stand in for the 5080's.

## Conventions

- No code comments or docstrings; use names, types and tests. Rationale belongs in these docs
  or commit messages.
- Product code in `chained_transformer/` never imports `tools/` or `tests/`.
- Everything large stays in ignored `artifacts/`. Compact evidence that a result needs is
  copied into `results/<name>/`.
- No hosted remote and no publishing until Philip says otherwise.

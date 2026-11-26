# chained-transformer

Research code for a Qwen3 transformer with one shared trunk and KV cache and several MLP
modes (think, code, answer) that the model switches between mid-answer. The
[README](README.md) states the idea, prior art and experiment plan.

Checkouts: `/home/philip/chained-transformer` on the RTX 5080 host and on the RTX 2080 Ti host
(SSH alias `pstack-2080`). Remote: `git@github.com:philippesic/chained-transformer.git`
(`origin`, public); both checkouts track `origin/main`. See
[development](docs/development.md#two-machines). Do not change its visibility or publish
elsewhere without asking.

## Navigation

- [Design](docs/design.md): terms (mode, position mode, segment label, trunk, arm, complete
  row) and how tracking, surgery, generation, data, training and evals work.
- [Development](docs/development.md): environment, check lanes, GPU sharing, conventions.
- [Data](docs/data.md): the data pipeline step by step, mix policies and every output field.
- [Tasks](tasks/README.md): unfinished-work convention. Active: [first experiment](tasks/first-experiment.md); [cloud/harness runbook](tasks/runbook-cloud-harness.md) for its first GPU runs.
- [FINDINGS](FINDINGS.md): index of results in `results/`.

## Code map

- `chained_transformer/modes.py`: the mode tracker; the single source of labelling rules for training and generation.
- `chained_transformer/surgery.py`: `ModeMLP`, `LoRAModeMLP`, `MoEMLP`, `Routing`, `apply_surgery`.
- `chained_transformer/generate.py`: the mode-tracking generation hook and the separate-models baseline.
- `chained_transformer/model.py`: base model loading and arm construction.
- `chained_transformer/data.py`, `chained_transformer/train.py`, `chained_transformer/evals.py`: entry points (`python -m ...`).
- `chained_transformer/mix.py`: per-row switching statistics and the data mix policies.
- `chained_transformer/optim.py`: `ChunkedAdamW8bit`. `chained_transformer/attention.py`: the fused-only attention. `chained_transformer/memory.py`: the training memory estimate.
- `tools/check.py`: the commit gate. `tools/gpu_equivalence.py`, `tools/gpu_smoke.py`, `tools/memory_check.py`: hardware lanes. `tools/memtrace.py`: CPU live-bytes tracker. `tools/mode_probe.py`: linear probes for mode on base hidden states.
- `docs/routing-and-looping.md`: design note on combining mode routing with looped layers.

## Rules

- Change labelling only in `ModeTracker`, then rebuild data slices; training labels and
  generation must use the same tracker.
- Run `python -m tools.check` before committing (the hook does it). Report CPU-only checks as
  CPU-only.
- GPU: the 5080 is reserved for BitDraft and is not used until Philip says so. On the 2080 Ti,
  queue this project's GPU jobs in pueue's `ct` group (`pueue add -g ct -- ...`); the `gpu` group
  there belongs to ShaderLOD and is never used. Inspect `nvidia-smi` and `pueue status` first;
  never stop another project's job. Jobs longer than a few minutes go through jobwatch; runs over
  about one GPU-hour need a preflight.
- Large outputs go to ignored `artifacts/`; record their purpose and disposal in the task.
- No code comments or docstrings. Commit small coherent checkpoints.

# chained-transformer

Research code for a Qwen3 transformer with one shared trunk and KV cache and several MLP
modes (think, code, answer) that the model switches between mid-answer. The
[README](README.md) states the idea, prior art and experiment plan.

Checkout: `/home/philip/chained-transformer` on the RTX 5080 host. A plain copy (rsync, not a
Git remote) sits at the same path on the RTX 2080 Ti host, SSH alias `pstack-2080`; it does not
sync itself. Local only: no remote, do not push or publish until Philip says so.

## Navigation

- [Design](docs/design.md): terms (mode, position mode, segment label, trunk, arm, complete
  row) and how tracking, surgery, generation, data, training and evals work.
- [Development](docs/development.md): environment, check lanes, GPU sharing, conventions.
- [Tasks](tasks/README.md): unfinished-work convention. Active: [first experiment](tasks/first-experiment.md).
- [FINDINGS](FINDINGS.md): index of results in `results/`.

## Code map

- `mode_mlp/modes.py`: the mode tracker; the single source of labelling rules for training and generation.
- `mode_mlp/surgery.py`: `ModeMLP`, `LoRAModeMLP`, `MoEMLP`, `Routing`, `apply_surgery`.
- `mode_mlp/generate.py`: the mode-tracking generation hook and the separate-models baseline.
- `mode_mlp/model.py`: base model loading and arm construction.
- `mode_mlp/data.py`, `mode_mlp/train.py`, `mode_mlp/evals.py`: entry points (`python -m ...`).
- `tools/check.py`: the commit gate. `tools/gpu_equivalence.py`, `tools/gpu_smoke.py`: hardware lanes.

## Rules

- Change labelling only in `ModeTracker`, then rebuild data slices; training labels and
  generation must use the same tracker.
- Run `python -m tools.check` before committing (the hook does it). Report CPU-only checks as
  CPU-only.
- GPU: the 5080 is shared with BitDraft. Inspect `nvidia-smi` and `pueue status` first; never
  stop another project's job. Jobs longer than a few minutes go through jobwatch; runs over about
  one GPU-hour need a preflight.
- Large outputs go to ignored `artifacts/`; record their purpose and disposal in the task.
- No code comments or docstrings. Commit small coherent checkpoints.

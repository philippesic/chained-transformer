# Serving notes

Work log for the serving path ([design](serving.md)): what is verified, where, and what is not.
Rewrite rather than append.

## Environment of the 2026-11-26 overnight cloud run

- CPU-only container: 4 Xeon cores (AVX-512, AMX), 15 GB RAM, no GPU.
- huggingface.co, ModelScope and download.pytorch.org were unreachable, so there were **no
  Qwen3-0.6B weights or tokenizer**. Every check used tiny random Qwen3 models. The two
  tokenizer tests in `tests/test_data.py` now skip, with the reason, when the tokenizer cannot
  be loaded; they still run wherever it is cached or downloadable.
- Every commit on the branch passed `tools.check` (the pre-commit hook) in the container.
- PyPI and `git clone` from GitHub worked.
- vLLM 0.34.0 was built for CPU (`VLLM_TARGET_DEVICE=cpu`, editable install) from the `v0.34.0`
  tag of its GitHub source, in a virtualenv outside the repo, with torch 2.14.0 and triton 3.8.0
  from PyPI. The CPU build of torch was out of reach, so torch's `Caffe2Config.cmake` in that
  virtualenv was edited to skip CUDA detection, and the build needed `libnuma-dev` from apt in
  the container.
- vLLM's engine on x86 CPU refuses to start without triton-cpu, which is published as a GitHub
  release wheel and not on PyPI. Installing it was declined in this run, so **vLLM's own engine
  never ran here**. The plugin was checked with a hand-driven engine instead (below).

## Verified

All on CPU, float32, tiny random Qwen3 models (2 layers, hidden 128, head_dim 32, 64-token
vocabulary) whose outputs depend on routing.

- **Batched mixed-mode reference** (`serving/mixed.py`, 21 tests): loop, gather,
  sort-and-segment and block-aligned implementations agree; batched decode with per-row trackers
  reproduces per-sequence generation token for token and mode for mode.
- **Triton grouped kernel** (`serving/triton_grouped.py`, 14 tests) under `TRITON_INTERPRET=1`
  against the loop reference. The interpreter shows correctness only, not GPU behaviour or speed.
- **Mode slots** (`serving/mode_slots.py`, 11 tests): chunked prefill, prefix-cache start,
  preempt and resume, spec-decode rollback by rebuild, dummy rows, slot reuse, unknown token ids.
- **Export** (`serving/export.py`, 4 tests): exact logits after a round trip, LoRA merged, and a
  single-mode export equals the chained model routed to that mode.
- **vLLM plugin against vLLM 0.34** (`tests/test_serving_vllm_parts.py`, 14 tests; they run in
  an environment with vLLM and skip in the repo's):
  - `ChainedQwen3ForCausalLM` registers, vLLM's registry resolves the exported checkpoint to it,
    and `hf_overrides` reach `mode_tracking`. The entry-point shim registers it when imported
    from its own directory.
  - vLLM's loader maps every checkpoint tensor into the model, per-mode gate and up merged into
    `gate_up_proj`; each mode's weights are views into the stacked tensors the Triton path uses.
  - Each MLP implementation (`select`, `loop`, `triton` through the interpreter) sends every
    token to its mode's MLP; dummy batches get mode 0 and leave trackers alone.
  - A hand-driven engine reproduces Hugging Face greedy decoding **token for token and mode for
    mode**. It uses the real vLLM model, vLLM's CPU paged attention with its metadata builder
    and KV cache layout, real `InputBatch` objects and `ModeModelState`; the test itself only
    decides which tokens run each step. Cases: 4 prompts, 24 new tokens, a 32-token budget
    (chunked prefill and mixed-mode decode steps) for `select`, `loop` and `triton`; 72-token
    prompts run twice with prefix-block reuse, so the second pass starts at position 64;
    preemption and resume mid-decode; and torch.compile at vLLM's CPU default
    (`DYNAMO_TRACE_ONCE` with inductor) for `select` and `triton`. Breaking routing (all tokens
    to mode 0, or mode ids shifted by one) makes every engine test fail.
  - To build a `VllmConfig` on x86 CPU without triton-cpu, the tests set vLLM's `HAS_TRITON`
    flag only while the config is created. Nothing else in vLLM is patched.
- **Paging** (`serving/paging.py`, 6 tests): LRU, refusing more modes than capacity, exact
  outputs, paged decode equal to resident decode at capacity 1 and 2, and a mixed batch over
  capacity streams the extra modes layer by layer.
- **CPU smoke runs** of `tools.bench_serving`, `tools.bench_mode_mlp` and `tools.bench_paging`
  (outputs in `artifacts/serving/*_cpu_tiny.json`), and `tools.bench_vllm prepare`. In
  `bench_serving`, chained batched decode matched chained per-sequence decode; separate
  fine-tunes with re-prefill made 5 switches and re-prefilled 67 tokens for 4 prompts. Timings
  on a 128-wide model on CPU say nothing about GPU performance.

## First GPU run

The checklist runs on the 2080 Ti (Turing, 11 GB, sm_75) in this project's pueue group `ct`
(parallelism 1). Pueue's `gpu` group belongs to ShaderLOD and is never used for this project;
the 5080 is reserved for BitDraft and is not used until Philip says so.

```sh
pueue add -g ct -- uv run --locked --no-sync python -m tools.serving_gpu_checklist
```

A preflight prints the device, compute capability, total and free memory, the torch, CUDA and
Triton versions in this env and the vLLM, torch and Triton versions in the vLLM virtualenv, and
a memory plan for each vLLM engine. It refuses to start, and says why, when there is no GPU,
the GPU is below sm_75, another process is on the GPU, or `--gpu-mem` asks for more than is
free. Then it runs `tools.check`, the Triton tests on the GPU, `tools.bench_mode_mlp` with its
CUDA graph check, the vLLM engine tests (eager, then compiled with CUDA graphs) in the vLLM
virtualenv, `tools.bench_vllm`, `tools.bench_paging` and `tools.bench_serving`. A vLLM test
that skips on the GPU counts as a failure.

Each step prints one line, `PASS`, `FAIL` or `SKIP`. A failed line names what broke, the likely
cause read from the log (out of memory, bfloat16 on Turing, shared memory, no kernel image,
vLLM's free-memory check, a missing model) and the likely fix. The lines and the preflight go
to `runs/serving-gpu-checklist-<time>.txt`, which is the file to paste back. Logs and
`report.md` stay in `artifacts/serving/gpu-checklist/<time>/`. `--only`, `--skip` and
`--model tiny` narrow the run; `--dry-run` prints the commands.

On the 2080 Ti:

- No bfloat16. Every step runs float16 or float32; vLLM refuses bfloat16 below sm_80 and the
  kernel refuses it with a message to serve in float16. Bfloat16 checks are 5080 only, once
  BitDraft is done and Philip says so.
- vLLM picks its Triton attention backend: FlashAttention needs sm_80 and vLLM 0.34 disables
  FlashInfer on sm_75.
- Triton runs single-stage loads (its software pipelining starts at sm_80) with 64 KiB of
  shared memory per block. A kernel-bench config that needs more is noted and not timed; the
  step fails only when no Triton config runs.
- Memory: Qwen3-0.6B takes 1.1 GiB in float16, the chained model with three modes 2.1 GiB, and
  the KV cache 0.11 GiB per 1,000 tokens. The defaults, `--gpu-mem 0.8` and
  `--max-model-len 1024`, leave each engine room, including the separate arm's three engines at
  a third of the budget each. The preflight's plan says which engine is short and the
  `--gpu-mem` that fits what is free.
- Speed numbers from the 2080 Ti are not 5080 numbers; 5080 numbers are 5080 only, once BitDraft
  is done and Philip says so.

`--allow-other-clients` starts next to another GPU process. A vLLM engine can then fail to
start if that process's memory use changes while vLLM profiles memory.

Changes made for that run, all checked on CPU:

- Float32 Triton dots use `input_precision="ieee"`. Triton defaults to TF32 on Ampere and newer,
  which would have failed the float32 kernel tests and the float32 vLLM test on the 5080. The
  2080 Ti has no TF32, so this matters there only once runs move to newer GPUs.
- The kernel checks shapes, dtypes, devices and block sizes before launching, and refuses
  bfloat16 below compute capability 8.0 with a message to serve in float16 (the 2080 Ti).
  `num_tokens` is not specialised, so new batch sizes do not recompile the kernel.
- The model raises when vLLM calls it without `mode_ids`, instead of decoding everything in
  `think`; the model state raises on modes outside `0..M-1` before they reach the GPU.
- `register()` names the vLLM version when Model Runner V2's model states are missing.
- `serving.vllm_plugin.state.runner_report` (for `collective_rpc`) reports the model state
  class, MLP impls, compile and CUDA graph modes, and whether the stacked weights still share
  memory with each mode's weights after vLLM's processing.
- vLLM gives each request an internal id (the caller's id plus `-` and 8 hex digits), so
  `collect_traces` keys traces by the caller's id; the engine tests looked them up by the
  caller's id and would have failed with KeyError.
- The engine tests set `TRITON_F32_DEFAULT=ieee`: float32 attention runs on vLLM's Triton
  backend, whose dots default to TF32 on Ampere and newer, which can flip exact greedy matches.
- The vLLM engine tests cap `gpu_memory_utilization` at 0.2 instead of vLLM's 0.9, run the
  Triton impl on CUDA, and `tests/test_serving_vllm_graphs.py` adds a compiled, CUDA-graph
  engine with the Triton impl checked against Hugging Face at three batch sizes.

## Serving a trained pilot checkpoint

The pilot saves each arm with `chained_transformer.checkpoint.save_trained` as a `chained_qwen3`
model. `full` checkpoints serve as saved: `register()` imports `chained_transformer.checkpoint`
so vLLM can read that config, and the weight names already match. `lora` checkpoints need
`serving.export.export_for_vllm`, which merges each mode's adapters; vLLM refuses them as saved
with a message saying so. `moe` checkpoints route by a learned router, not by mode, and are not
served. Exports now write `model_type` `qwen3`: they used to keep `chained_qwen3`, which a fresh
vLLM process could not read, so every exported pilot checkpoint would have failed to load.
`tools.bench_vllm prepare --checkpoint PATH` (and the checklist's `--checkpoint`) benchmarks a
trained arm; its `base` arm is then the checkpoint's `think` mode on its own.

Checked on CPU with tiny `full` and `lora` pilot checkpoints: both, exported, and the `full` one
as saved, decode token for token and mode for mode like Hugging Face in the hand-driven engine,
and a fresh process with only `register()` reads both the export and the saved `full` checkpoint.

## Not verified

- vLLM's own engine driving the plugin: its scheduler, GPU model runner and sampler calling
  `add_request`, `remove_request`, `prepare_inputs` and `prepare_dummy_inputs`.
  `tests/test_serving_vllm.py` checks exactly that and needs a GPU (or triton-cpu).
- `VLLM_COMPILE` piecewise compilation and CUDA graph capture and replay with `mode_ids`. The CPU
  platform forces `DYNAMO_TRACE_ONCE` and has no CUDA graphs.
  `tests/test_serving_vllm_graphs.py` checks this on a GPU.
- The Triton kernel on a GPU: correctness in compiled mode, block sizes and speed on the 2080 Ti,
  and on the 5080 (5080 only, once BitDraft is done and Philip says so).
- Loading the plugin in vLLM worker processes through the `vllm.general_plugins` entry point.
  The shim was imported directly; it was not installed.
- Real Qwen3 checkpoints and tokenizer, including tracker texts from a real tokenizer and
  thinking prompts. With `--model qwen` the benches copy the base MLP into every mode
  (`--perturb` adds noise) unless `tools.bench_vllm` gets `--checkpoint`; no real trained
  checkpoint has been served.
- Every GPU number: `tools.bench_vllm` arms, `tools.bench_mode_mlp` on CUDA with its CUDA graph
  check, `tools.bench_paging` transfer rates and overlap, `tools.bench_serving --model qwen`.
- Speculative decoding through vLLM. Rollback is covered by rebuild in the slot unit tests only.
- That stacked weights cost no extra GPU memory. vLLM's CUDA path leaves linear weights in place
  after loading, so the per-mode views should survive, but no GPU run measured it.
- llama.cpp: design sketch only ([llama-cpp.md](llama-cpp.md)), no code.

## Known limits

- `loop` is eager only: its data-dependent shapes fail under torch.compile. `tools.bench_vllm`
  forces `enforce_eager` for it.
- `select` runs every mode's MLP on every token, so it costs `M` times the MLP compute; it is
  the safe default, not the fast path.
- The tracker needs each step's input tokens on the host: one device-to-host copy per step (see
  the design's device-side tracker). The copy waits for the GPU, which cancels vLLM's async
  scheduling overlap, so the chained arms' tokens per second understate the plugin.
- On CPU, vLLM repacks linear weights after loading, so the stacked copy the Triton path keeps
  is extra memory there. Triton is a GPU path, so this only matters for tests.
- The `grouped` and `fused_moe` implementations in the design were not built into the plugin;
  `serving/mixed.py` has the PyTorch grouped reference.

## Blocked or skipped

- triton-cpu: install declined, so vLLM's engine could not start; replaced by the hand-driven
  engine tests.
- Hugging Face Hub unreachable: no Qwen weights or tokenizer.
- The two tokenizer tests skip in the container (no tokenizer), so tokenization against the real
  Qwen3 vocabulary was not exercised there.
- The entry-point shim (`serving/vllm_entry`) was not pip-installed.

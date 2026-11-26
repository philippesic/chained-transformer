# The vLLM plugin

`serving/vllm_plugin` serves a chained model, Qwen3 with one MLP per mode, from one vLLM engine
and one KV cache. For each request it runs the same `ModeTracker` that labels the training data
over the request's tokens on the host, and passes every forward a `mode_ids` tensor that picks
each token's MLP. Attention and the KV cache are shared, so a mode switch costs nothing beyond
routing. [serving.md](serving.md) has the design and [serving-notes.md](serving-notes.md) what is
verified; nothing has run on a GPU yet.

It is built on vLLM 0.34.0 and its default GPU runner (Model Runner V2); `register()` refuses a
vLLM without that runner.

## Requirements

- vLLM 0.34.0 in its own virtualenv, outside `uv.lock` (`MORNING.md` step 2), with this repo on
  `PYTHONPATH`.
- A GPU at sm_75 or newer.
- A model exported for vLLM (below), or a pilot `full` checkpoint as the pilot saved it.

## Export a model

`serving.export.export_for_vllm` writes `config.json` (architecture `ChainedQwen3ForCausalLM`,
`model_type` `qwen3`, a `mode_tracking` block) and `model.safetensors` with each mode's MLP under
`model.layers.N.mlp.modes.M`. Save the tokenizer next to it; the tracker reads token texts from
it.

```python
from pathlib import Path

import torch

from chained_transformer.checkpoint import load_checkpoint
from chained_transformer.model import load_tokenizer
from serving.export import export_for_vllm

built = load_checkpoint(Path("PATH/TO/CHECKPOINT"), dtype=torch.float32)
out = export_for_vllm(built.model.eval(), Path("artifacts/serve/ARM"), dtype=torch.float16)
load_tokenizer("Qwen/Qwen3-0.6B").save_pretrained(out)
```

- `full` checkpoints also serve as saved. `lora` checkpoints must be exported: the export merges
  each mode's adapters into a full MLP (`W + (alpha/r) B A`), and vLLM refuses a saved `lora`
  checkpoint with a message saying so. `moe` checkpoints route by a learned router and are not
  served.
- `export_single_mode` writes one mode as a plain Qwen3 model, which is what the benchmarks'
  `separate` arm serves.
- `python -m tools.bench_vllm prepare --dir DIR` does all of this for the base model or a
  `--checkpoint`, and is the quickest way to get a servable folder (see
  [demo-mode-swap.md](demo-mode-swap.md)).

## Load the plugin

vLLM must know the architecture in every process that builds the model. There are two ways.

**In the same process**, which is what the tests and benchmarks do. Turn off vLLM's engine
process so the engine runs where `register()` ran:

```python
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from serving.vllm_plugin import register

register()

from vllm import LLM, SamplingParams

llm = LLM(
    model="artifacts/serve/ARM",
    dtype="float16",
    enable_prefix_caching=True,
    hf_overrides={"mode_tracking": {"prompt_mode": 0, "mlp_impl": "triton"}},
)
outs = llm.generate(
    ["Write a Python function that reverses a string."], SamplingParams(max_tokens=256)
)
```

Run it from the repo root with `env PYTHONPATH=$PWD ~/venvs/vllm/bin/python script.py`.

**Through vLLM's plugin entry point**, for `vllm serve` or with the engine process on.
`serving/vllm_entry` is a one-file package whose `vllm.general_plugins` entry point calls
`register()` in every vLLM process. It finds the repo from its own location, or from
`CHAINED_TRANSFORMER_ROOT`. Installing it puts a package into the vLLM virtualenv, so like the
virtualenv itself it's your call:

```sh
uv pip install --python ~/venvs/vllm/bin/python -e serving/vllm_entry
CHAINED_TRANSFORMER_ROOT=$PWD ~/venvs/vllm/bin/vllm serve artifacts/serve/ARM --dtype float16
```

Neither the install nor `vllm serve` has been tried; the shim was only imported directly.

## Settings: `mode_tracking`

The export writes a `mode_tracking` block into `config.json`. Change it there, or per engine with
`hf_overrides={"mode_tracking": {...}}`. vLLM sets a dict override as a whole, so the override
replaces the block: repeat every key you rely on.

| Key | Default | Meaning |
|---|---|---|
| `mlp_impl` | `select` | how a batch runs its modes' MLPs: `select`, `loop` or `triton` (below) |
| `prompt_mode` | the checkpoint's prompt mode, else `0` (think) | mode of tokens outside an assistant turn; `0` think, `1` code, `2` answer |
| `token_texts_file` | none | a JSON list of each token id's text, in the model folder; without it the tracker decodes the tokenizer's vocabulary once at start |
| `record` | `false` | keep every request's per-token modes for `collect_traces`; for tests, memory grows with tokens |

## Choosing `mlp_impl`

| Impl | What it does | Use it when |
|---|---|---|
| `select` | runs every mode's MLP on every token and keeps the right one: three times the MLP compute | the safe default; the only one for quantized weights or tensor parallelism |
| `loop` | runs each mode's MLP on that mode's rows | eager engines only; its shapes depend on the data, so `torch.compile` fails on it |
| `triton` | one grouped Triton kernel, each token through its own mode only, with static shapes so compilation and CUDA graphs work | the fast path, once the serving checklist's `triton-gpu` and `vllm-graphs` steps pass on the card |

`triton` reads stacked copies of the weights made after loading. `runner_report` (below) lists
any layer where vLLM replaced a weight after that, which would leave `triton` reading stale
weights.

On an engine started with `enforce_eager=True`, `llm.collective_rpc(set_mlp_impl,
args=("triton",))` switches the impl in place; it returns the number of mode MLPs changed. On a
compiled or graph-captured engine the impl is built in, so it raises; start a new engine with the
impl in `hf_overrides` instead.

## Per GPU

| GPU | dtype | Notes |
|---|---|---|
| 2080 Ti (sm_75) | `float16`, or `float32` | No bfloat16: vLLM refuses it below sm_80 and the Triton kernel refuses it too. vLLM uses its Triton attention backend. Qwen3 may overflow float16; step 2 of the demo checks it. |
| 5080 | `bfloat16` | 5080 only, once BitDraft is done and Philip says so. |

## What works with it, and what it refuses

Works by design, checked against vLLM's code and with CPU tests, not yet on a GPU:

- **Prefix caching.** Cache blocks are keyed by tokens, and a token's mode depends only on the
  tokens before it, so cached blocks stay valid. That holds while the tracker settings stay the
  same; a cache shared across engines with different `prompt_mode` or token texts would not be.
- **Chunked prefill and preemption.** Chunks are fed in order; a preempted request is re-seeded
  from its prompt and the tokens it had.
- **CUDA graphs and `torch.compile`**, with `select` and `triton`.
- **Async scheduling** (on by default). Correct, but the tracker copies each step's tokens to the
  host before the forward, which stalls the overlap async scheduling is for. A device-side
  tracker would remove that; it is not built.

Refused at engine start, with a message naming the feature, because the plugin would route
tokens wrong without any error:

- speculative decoding (draft tokens reach the tracker before they are accepted);
- micro-batching (`--enable-dbo`), which splits `input_ids` per micro-batch but not `mode_ids`;
- pipeline parallelism;
- `triton` with tensor parallelism or quantized weights.

Untested and not refused: tensor parallelism and quantized weights with `select` or `loop`,
vLLM's own LoRA adapters on top, and `vllm serve`.

## Inspecting a running engine

All three take the engine's worker and run through `llm.collective_rpc`:

```python
from serving.vllm_plugin.state import collect_traces, runner_report, set_mlp_impl

report = llm.collective_rpc(runner_report)[0]
stats = llm.collective_rpc(collect_traces)[0]
```

- `runner_report`: vLLM version, the model state class (should be `ModeModelState`), number of
  mode MLPs, impls in use, `stacking_problems`, the weights' device and dtype, compilation and
  CUDA graph modes, and `enforce_eager`.
- `collect_traces`: `steps`, `mixed_steps` (steps whose batch held more than one mode), tracker
  `rebuilds`, and, with `record` on, each request's modes per token under `traces`.

## Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| `Model architectures ['ChainedQwen3ForCausalLM'] are not supported for now` | the plugin is not registered in the process that built the model | call `register()` with `VLLM_ENABLE_V1_MULTIPROCESSING=0`, or install the entry point |
| `serving.vllm_plugin was built on vLLM 0.34.0's Model Runner V2, but vLLM ...` | another vLLM version | use vLLM 0.34.0 |
| `ChainedQwen3ForCausalLM got no mode_ids` | a model runner that does not call the plugin's `ModeModelState` | vLLM 0.34 with its default GPU runner |
| `... does not support speculative decoding` (or micro-batching, pipeline parallelism) | see above | drop that option |
| `mlp_impl 'triton' does not all-reduce across tensor-parallel ranks` or `reads unquantized stacked weights` | see above | one GPU, or `select` |
| `bfloat16 tl.dot needs compute capability 8.0 or newer` | bfloat16 on the 2080 Ti | `dtype="float16"` |
| `vLLM serves full per-mode MLPs; export the 'lora' checkpoint ...` | a saved `lora` (or `moe`) checkpoint | export it first; `moe` is not served |
| `mode weights were not stacked after loading`, or `stacking_problems` in `runner_report` | vLLM changed the weights after loading, for example to quantize or repack them | `select` |
| `this engine was compiled or captured CUDA graphs with its mlp_impl built in` | `set_mlp_impl` on a compiled engine | a new engine with the impl in `hf_overrides`, or `enforce_eager=True` |
| `mode tracking produced N modes for M tokens` | a bug in the tracker's bookkeeping | report it with the prompt; `record=True` and `collect_traces` show the modes so far |
| float16 outputs that repeat one token from the start | float16 overflow on the 2080 Ti | `float32`; check with `tools.compare_outputs` |

# Serving

How chained-transformer is meant to be served, what vLLM gives us for it, and what is built
and verified. The claim the serving path has to support: **a request switches mode mid-answer
on one shared KV cache**, while the "separate fine-tunes" alternative must re-prefill the
history into another model's cache at every switch. Work log and unverified items are in
[serving notes](serving-notes.md).

All vLLM references are to the v0.34.0 tag (`git clone --branch v0.34.0
https://github.com/vllm-project/vllm`), read on 2026-11-26.

## What vLLM 0.34 gives us

### Out-of-tree models

- `ModelRegistry.register_model(arch, "module:Class")` (`vllm/model_executor/models/registry.py`
  ~1255) registers a lazily imported class under an `architectures` name from `config.json`.
- Plugins are entry points in the `vllm.general_plugins` group (`vllm/plugins/__init__.py`).
  `load_general_plugins()` runs in the front-end process, the engine core
  (`vllm/v1/engine/core.py`) and every worker (`vllm/v1/worker/worker_base.py`), so a plugin must
  be an installed distribution, not an in-process call, unless everything runs in one process
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, uniprocess executor).
- The in-tree `Qwen3ForCausalLM` hard-codes `self.model = Qwen3Model(...)`, `Qwen3Model` builds
  layers from `decoder_layer_type`, and `Qwen3DecoderLayer.forward(positions, hidden_states,
  residual)` calls `self.mlp(hidden_states)`. The MLP is `Qwen2MLP`: one
  `MergedColumnParallelLinear` `gate_up_proj`, `SiluAndMul`, one `RowParallelLinear`
  `down_proj`. `qwen3_vl.py`'s `Qwen3LLMModel` is the in-tree precedent for subclassing the
  decoder and adding a forward argument.
- Weight loading is `AutoWeightsLoader` plus `Qwen2Model.hf_to_vllm_mapper`, which rewrites the
  substrings `.gate_proj`/`.up_proj` to `.gate_up_proj` shards 0/1. A checkpoint named like our
  Hugging Face surgery (`model.layers.{i}.mlp.modes.{m}.gate_proj.weight`) loads without a custom
  loader if the vLLM module tree is `mlp.modes.{m}.gate_up_proj`.

### Per-request state: Model Runner V2 `ModelState`

v0.34 has one model runner, "Model Runner V2" (`vllm/v1/worker/gpu/model_runner.py`); the CPU
runner subclasses it on x86 (`vllm/v1/worker/cpu/model_runner.py`). A model can supply its own
`ModelState` through `get_model_state_cls()` (`vllm/v1/worker/gpu/model_states/__init__.py`).
The interface (`model_states/interface.py`) has exactly the hooks mode tracking needs:

| Hook | When | Use for modes |
| --- | --- | --- |
| `add_request(req_index, new_req_data)` | new or resumed request, before the forward | reset the slot's tracker; on resume `prefill_token_ids` holds prompt plus generated tokens |
| `remove_request(req_id)` | finished or preempted | drop the slot |
| `prepare_inputs(input_batch, req_states) -> dict` | every real step, before the forward | feed each request's scheduled tokens to its tracker, fill a persistent `mode_ids` buffer, return it as a forward kwarg |
| `prepare_dummy_inputs(num_reqs, num_tokens) -> dict` | warm-up and CUDA-graph capture | return the same persistent buffer, so captured graphs read it |

`InputBatch` gives everything needed to map flattened tokens to requests: tokens are contiguous
per request in `req_ids` order, `query_start_loc_np` delimits them, `idx_mapping_np` gives each
row's persistent slot, `num_computed_tokens_np` gives the first scheduled position, and
`input_ids` holds the tokens (decode inputs are the previous step's samples, written by
`combine_sampled_and_draft_tokens` before `prepare_inputs`). `LongcatNgramModelState`
(`model_executor/models/longcat_flash_ngram.py`) is the in-tree analog: per-slot token history
feeding a persistent buffer. `prepare_inputs` also runs on dummy batches with made-up `req_ids`
and `idx_mapping = arange`, so a state must check slot ownership before touching a tracker.

`ForwardContext` and the attention metadata carry no request ids, and there is no global
accessor for the runner, so `ModelState` is the only clean hook. Custom logits processors
(`vllm/v1/worker/gpu/sample/logits_processor/`) see per-request token history too, but their
`apply` runs after the forward of the same step, so they cannot set that step's modes.

### torch.compile and CUDA graphs

`Qwen3Model` is wrapped in `@support_torch_compile(dynamic_arg_dims=...)`: one trace with no
guards, only listed arguments dynamic. CUDA graphs default to `FULL_AND_PIECEWISE`; piecewise
splits at attention, so MLPs run inside captured pieces, and FULL replays with no Python at all.
Consequences for a per-token mode id:

- It must be a forward argument (added to `dynamic_arg_dims` with dim 0), not a module
  attribute read during the forward, or the trace bakes it in.
- It must live in a persistent buffer refilled in place and passed sliced `[:num_tokens]`, also
  from `prepare_dummy_inputs`; replay reads the captured address.
- The MLP must not branch on data or produce data-dependent shapes (`nonzero`, `.item()`,
  per-mode Python loops over token counts). The repo's `_route` does all three, so it only
  works eagerly.

LoRA solves the same problem today: `token_lora_mapping` is the request's adapter id repeated
over its scheduled tokens, copied into the persistent `_token_lora_indices` of the Punica
wrapper (`vllm/lora/punica_wrapper/punica_base.py`), and the kernels gather per token. The
kernels are per token already; only the mapping is per request.

On the CPU platform (`vllm/platforms/cpu.py`) there are no CUDA graphs, `VLLM_COMPILE` becomes
`DYNAMO_TRACE_ONCE` with inductor, attention is `CPU_ATTN`, async scheduling is off, and x86
requires Triton with its CPU backend (`triton-cpu`).

### Prefix caching

Block hashes chain `(parent_hash, block_token_ids, extra_keys)` (`hash_block_tokens`,
`vllm/v1/core/kv_cache_utils.py`), with extra keys from LoRA name, multimodal inputs,
`cache_salt` and prompt embeddings. A token's K/V depends on the modes of every earlier
position (the MLP output feeds the next layer's attention), but each position's mode is a
deterministic function of the token prefix and the tracker's configuration. So **token-keyed
hashing stays correct with no extra key**, as long as the tracker configuration is fixed per
served model. If tracking ever becomes per request (for example a request-level `prompt_mode`
or a forced mode), that setting must enter the hash, the way LoRA's name does. Note that this
is the property separate fine-tunes lack: their caches are not interchangeable, which is why
they re-prefill.

### Fused MoE

`fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, ...)`
(`vllm/model_executor/layers/fused_moe/fused_moe.py`) runs a grouped GEMM after
`moe_align_block_size` sorts token ids by expert and pads each expert's run to the block size.
Hard routing is top-1 MoE with `topk_ids = mode_ids[:, None]` and `topk_weights = 1`. On CPU,
`cpu_fused_moe(...)` (`vllm/_cpu_ops.py`) does the same for unquantized weights whose sizes are
multiples of 32 (AMX variant bf16 only). Our modes store `gate_proj`, `up_proj`, `down_proj`
per mode, which stack into MoE's `w13 [M, 2I, H]` and `w2 [M, H, I]` layout.

## Design

### Model: `ChainedQwen3ForCausalLM`

- `config.json` keeps `model_type: qwen3`, sets `architectures: ["ChainedQwen3ForCausalLM"]` and
  keeps the `mode_surgery` block that `apply_surgery` already writes, plus a `mode_tracking`
  block (`prompt_mode`, optional `token_texts_file`, `mlp_impl`, and `record` to keep per-request
  mode traces for tests).
- Each decoder layer's MLP becomes a module with `modes = ModuleList[Qwen3MLP]`, so the surgery
  checkpoint loads as is. Its forward takes `(x, mode_ids)` and dispatches to one implementation:
  - `select`: every mode's MLP on every token, picked per row with `torch.where`. Static shapes,
    so it works under torch.compile and CUDA graphs; costs `M` times the MLP compute. Default.
  - `loop`: per-mode gather, MLP, scatter. Eager only. The reference.
  - `triton`: the grouped kernel below, registered as the custom op
    `chained_transformer::grouped_mode_mlp` so torch.compile treats it as opaque; static grid,
    no host sync. Mode weights are stacked once after loading and each mode's weights become
    views into the stack, so nothing is duplicated.
  - Not built: `grouped` (stable sort by mode, per-mode GEMM on contiguous runs, unsort; the
    PyTorch version is in `serving/mixed.py`) and `fused_moe` (vLLM's `fused_experts` or
    `cpu_fused_moe` with top-1 ids).
- `full` checkpoints serve natively. `lora` checkpoints are merged at export
  (`W_m = W + (alpha/r) B_m A_m`), which loses LoRA's memory advantage and is equal only up to
  rounding; a native LoRA path is shared base GEMMs for all tokens plus a grouped low-rank delta
  per mode (Punica/S-LoRA SGMV with a per-token index), and is future work. `moe` is a training
  baseline with learned soft routing and is not served by this plugin.

### Mode tracking per request: `ModeModelState`

- One `ModeTracker` per persistent slot, the same class used for training labels, fed token
  strings decoded once per vocabulary id (`TokenTexts`). Using the training tracker is what keeps
  serving routing identical to training routing.
- Each slot records its owner `req_id` and `fed`, the number of tokens already fed. In
  `prepare_inputs`, for every row whose `req_id` owns its slot: if `fed` is past the row's
  first scheduled position, rebuild the tracker and re-feed the tokens the slot already knows
  (its prefill tokens and every token fed since); if it is behind (prefix-cache hit), feed the
  known tokens up to that position. Then feed the row's scheduled tokens and write their
  position modes. Rows that do not own their slot (dummy runs) get mode 0 and leave trackers
  alone. The rebuild was meant for spec-decode rejection, but vLLM's spec decode hands the host
  tokens before they are accepted, so the plugin refuses speculative decoding at engine start
  (with micro-batching and pipeline parallelism, which it also cannot route); the rebuild stays
  as a guard.
- Position mode of token `p` is the tracker state after reading token `p`, matching the
  Hugging Face hook and the training labels.
- Chunked prefill feeds chunks in order; resume re-seeds from `prefill_token_ids`.

Cost: on GPU, feeding the tracker needs this step's input tokens on the host, so every step
copies `input_ids[:num_tokens]` to the CPU before the forward. That is one sync per step and it
defeats async scheduling's overlap. The fix is a device-side tracker: the character automaton
has bounded state (mode, outer mode, fence length, thinking-done flag, a line-shape state, and
an Aho-Corasick state for `</think>`, `<|im_start|>`, `<|im_end|>`), so a small kernel can walk
each new token's bytes from a `[vocab, max_len]` byte table on device. It must be
differential-tested against `ModeTracker` on random and real traces before it replaces it.

### Mixed-mode batches as hard-routed top-1 MoE

A decode batch holds requests in different modes. Running each mode's MLP on its rows is a
grouped GEMM, the same problem as MoE with one expert per token:

1. Align: stable-sort token indices by mode, pad each mode's run to `BLOCK_M`, record each
   block's mode. With `T` tokens and `M` modes the padded length is at most
   `T + M * (BLOCK_M - 1)`, so every buffer has a static size and no host sync is needed.
2. Gate-up kernel: each program takes one block of same-mode rows and one tile of the
   intermediate dimension, gathers the rows, multiplies by that mode's `gate` and `up` tiles in
   two accumulators, and writes `silu(g) * u` in sorted order.
3. Down kernel: contiguous sorted rows times that mode's `down` tile, scattered back to the
   original token positions; padding rows are skipped.

Compared with one dense MLP, the weight traffic is one MLP per *distinct* mode in the batch,
which is what decode (memory-bound) pays. When every token is in one mode it degenerates to the
dense case plus a sort of `T` ids. The scheduler could also sort the batch by mode so runs are
contiguous already; that is an optimisation, not a requirement.

### Paging idle modes to CPU

Qwen3-0.6B's MLP is about 264M parameters per mode, 0.53 GB in bf16, so three resident modes fit
any GPU and paging buys nothing at this size. It matters for more modes, larger bases and
single-user desktop runtimes (llama.cpp). A batch needs every mode that any of its tokens uses
resident, so paging works at mode granularity with LRU eviction, and the scheduler should group
requests by mode when memory is short. A switch is known one step ahead: the token that triggers
it (`</think>`, the newline that ends a fence line) is sampled one step before the first token
that uses the new mode, so the load can start then and stream layer by layer, overlapping layer
`l`'s transfer with layers before it. At PCIe 3 (the 2080 Ti) one 0.6B mode is roughly 45 ms,
several decode steps, so un-overlapped paging would be visible on every switch.

## Status

See [serving notes](serving-notes.md) for what is verified, where, and what is not.
[serving-plugin.md](serving-plugin.md) is the guide to using the plugin.
[demo-mode-swap.md](demo-mode-swap.md) has the commands for the one-cache demo against
separate models on the 2080 Ti and the 5080.

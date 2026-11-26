# Serving chained modes in llama.cpp: design sketch

Status: design only, nothing here is built or tested. File and symbol names were read from
llama.cpp at commit `28148b5` (2026-11-26); they move often, so check them again before starting.

## The short version

A chained model is a top-1 MoE whose router is replaced by the mode tracker. llama.cpp already
has every piece except the tracker:

- Mode MLPs stored as MoE expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, shape
  `[n_embd, n_ff, n_mode]`), so `ggml_mul_mat_id` picks each token's mode in a mixed batch.
- `llm_graph_context::build_moe_ffn` takes `selected_experts_in` and `probs_in`
  (`src/llama-graph.cpp`), so the graph can take the expert ids from an input tensor instead of
  a router.
- Idle modes can live in host memory with `--cpu-moe` / `--n-cpu-moe` (override regex
  `LLM_FFN_EXPS_REGEX` in `common/common.h`), and `--moe-cache-mib` keeps recently used experts
  in a GPU cache (`src/llama-moe-cache.{h,cpp}`). With one active mode per sequence, that is the
  mode paging from [serving.md](serving.md) without new code.
- The KV cache is shared by all modes, so nothing changes there.

What is missing is a way to get the per-token mode into the graph, and the tracker that
computes it.

## Pieces

1. **Conversion** (`convert_hf_to_gguf.py`, arch modules in `conversion/`). Add a converter for
   `ChainedQwen3ForCausalLM` (the export from `serving/export.py`): Qwen3's tensors, plus the
   per-mode MLPs stacked into `ffn_{gate,up,down}_exps` the way the Qwen3-MoE converter stacks
   experts. Metadata: `<arch>.expert_count = n_modes`, `<arch>.expert_used_count = 1`, and
   `<arch>.mode_prompt` (the tracker's prompt mode). Merge LoRA before converting, as for vLLM.
2. **Architecture** (`src/llama-arch.{h,cpp}`, `src/models/`). A new `LLM_ARCH_QWEN3_CHAINED`
   copying `src/models/qwen3.cpp` for tensors and attention, with the MLP replaced by
   `build_moe_ffn(..., selected_experts_in = inp_mode, probs_in = inp_mode_probs)`. To get a weight
   of exactly 1, either pass one-hot logits (0 for the mode, `-INFINITY` elsewhere) with softmax
   gating, or add a gating type that passes the weights through.
3. **Graph input.** An `llm_graph_input_mode` like `llm_graph_input_pos`: an `I32 [1, n_tokens]`
   tensor filled in `set_input` from the ubatch. All layers share it.
4. **Where modes come from.** `llama_batch` (`include/llama.h`) carries token, embd, pos, seq ids
   and logits flags, but no history, and the KV cells keep positions, not tokens. Two options:
   - API: add an optional `int8_t * mode` array to `llama_batch` (NULL means "track it for me"),
     and let applications that know better fill it.
   - Library: keep one tracker per `llama_seq_id` in the context, fed from
     `llama_token_to_piece` as batches are decoded. It must rewind when the application removes
     KV cells (`llama_memory_seq_rm`) or reuses a cached prompt prefix: keep the token ids per
     sequence and replay, as `serving/mode_slots.py` does for vLLM.
   The second keeps every existing client working; the first is simpler to get right. A small C
   port of `ModeTracker` (`chained_transformer/modes.py`, about 100 lines) is needed either way.
5. **Server.** `tools/server` slots already keep their cached tokens and call `seq_rm` on cache
   reuse, so slot-level trackers can replay the kept prefix. Mixed-mode batches across slots need
   nothing extra: `ggml_mul_mat_id` takes a different id per token.
6. **Tests.** Export the tiny model from `serving/tiny.py`, convert it, and compare greedy tokens
   and per-position modes with the HF reference, as `tests/test_serving_vllm.py` does for vLLM.

## Costs and limits

- Decode reads one mode's MLP per active mode in the batch; a single user reads one. Prefill of a
  prompt that spans modes reads every mode it uses once per ubatch.
- With `--cpu-moe`, the CPU computes the active mode's MLP from host memory each token, so the
  cost is one mode's MLP bandwidth on the CPU, not three. With `--moe-cache-mib` large enough for
  one or two modes per layer, the active modes stay on the GPU after the first token of a run.
- Speculative decoding drafts tokens in the current mode; a draft that crosses a mode switch is
  verified with modes computed from the drafted tokens, so the tracker must run on draft tokens
  too and rewind on rejection.
- Grammar, sampling and chat templates are unaffected.

## Order of work

Converter and arch with modes supplied by the caller (option 1 of piece 4) first, checked
against HF on the tiny model; then the in-context tracker; then the server and cache flags.

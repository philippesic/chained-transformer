# Design

How the maintained code works now. Open choices live in the
[task](../tasks/first-experiment.md); measured conclusions live in [FINDINGS](../FINDINGS.md).

## Terms

- **Mode**: one MLP set per layer. `think` = 0, `code` = 1, `answer` = 2 (`chained_transformer/modes.py`).
- **Position mode**: the mode whose MLP runs at a position. It is the tracker state *after*
  reading that position's token, so it is the mode that produces the next token.
- **Segment label**: the mode that produced a token, which is the position mode of the token
  before it. Losses are grouped by segment label.
- **Trunk**: everything shared: embeddings, attention, norms, LM head (tied to embeddings).
- **Arm**: one of the five training setups (`sft`, `full`, `lora`, `moe`, `separate`).
- **Complete row**: a training row that contains `</think>` and has fewer than 16,000
  assistant tokens (see [data](#data)).

## Mode tracking (`chained_transformer/modes.py`)

`ModeTracker` is a character-level state machine with no lookahead. Training labels and
generation use the same tracker, fed token strings in order, so a model trained on these labels
sees the same routing when it generates. A test checks that the result does not depend on where
token boundaries fall.

Rules:

- Outside an assistant turn (system or user text) the mode is `prompt_mode`, default `think`.
  Fences and tags in the prompt do not switch modes.
- `<|im_start|>assistant\n` starts an assistant turn in `think`.
- The first `</think>` of a turn switches to `answer` and closes any open fence. Later
  `</think>` strings are plain text. A trace without `</think>` stays in `think`.
- A line made of optional indentation, three or more backticks and an optional info string
  without backticks opens a fence. The switch to `code` happens at the end of that line, so the outer mode
  writes the fence line. A line holding only at least as many backticks closes it, again at the
  end of the line; then the mode returns to whatever opened the fence (`think` or `answer`).
- Four-backtick fences contain three-backtick lines; a line like ```` ```python ```` inside a
  fence does not close it. Inline triple backticks and `~~~` fences are not tracked.
- An unclosed fence in the answer stays `code` until `<|im_end|>`.

## Model surgery (`chained_transformer/surgery.py`)

`apply_surgery(model, SurgeryConfig)` replaces every Qwen3 decoder layer's `mlp` and leaves
attention and the KV cache untouched, which is why one cache serves every mode. Variants:

| Variant | Module | Parameters (Qwen3-0.6B) |
| --- | --- | --- |
| `full` | `ModeMLP`: N deep copies of the base MLP | 1,124.5M |
| `lora` | `LoRAModeMLP`: shared base gate/up/down plus per-mode LoRA (rank 64, alpha 128, B = 0) | 662.1M |
| `moe` | `MoEMLP`: E copies as experts, linear router, top-k softmax renormalised, Switch-style balance loss | 1,124.6M |

All three start equal to the base model: copies are identical, LoRA B is zero, and MoE weights
sum to one over identical experts. `MoEMLP` routes in fp32 and accumulates the weighted expert
outputs in an fp32 buffer, casting to the model dtype once at the end. Combining in bf16 (the
first version) rounded each weight, each product and each sum, so identical experts did not
add back to the base output: on the 2080 Ti the bf16 `moe` arm's logits drifted 0.4 to 0.66 from
the base model's while `full` matched exactly.

The per-token mode index reaches the MLPs through a shared `Routing` object returned by
`apply_surgery`: call `routing.set(mode_ids)` with a `[batch, seq]` tensor before each forward.
The value stays set through backward, so activation-checkpoint recomputation sees the same
routing. When every token has the same mode, the MLP runs on the whole tensor (the decode
path); otherwise each mode gathers its rows and the results are scattered back. A missing or
wrongly shaped `mode_ids` raises an error naming the repair.

`freeze_all_but_mode(model, m)` leaves only mode `m`'s MLP (or its LoRA tensors) trainable,
for single-mode retraining.

## Checkpoints (`chained_transformer/checkpoint.py`)

A mode checkpoint is a normal Hugging Face directory (`config.json`, `model.safetensors`,
`generation_config.json`, tokenizer files) whose config has `model_type: chained_qwen3`.
`ChainedQwen3Config` extends `Qwen3Config` with `mode_surgery` (the `SurgeryConfig`),
`mode_names`, `mode_switch` (the tracker's switch strings, fence rule, prompt mode and a
tracker version) and `base_model`. `ChainedQwen3ForCausalLM` is a `Qwen3ForCausalLM` that
applies the surgery in its constructor and exposes `model.routing`. Importing the module
registers both with `AutoConfig` and `AutoModelForCausalLM`, so `save_pretrained` and
`from_pretrained` handle safetensors, tied embeddings and dtype. `load_checkpoint` returns a
`Built` for chained and plain Qwen3 directories alike (plain ones have no routing).

`convert(base, surgery)` applies the surgery to a loaded base model, builds the chained model on
the meta device and assigns the weights, so every mode starts as an exact copy of the base MLP,
LoRA B is zero and the MoE router is freshly initialised from `seed`. Non-persistent buffers
(rotary frequencies) are copied from the base and the LM head is re-tied.

On CPU in fp32, a fresh conversion reproduces the base logits bitwise for `full` and `lora` when
all tokens share a mode; mixed-mode batches differ by about 4e-6 (row-gathered matmuls), and
`moe` by about 3e-6 (weighted sum of identical experts). `tools/conversion_check.py` measures this
for prefill, cached decoding across switches and a save/load round trip.

`QWEN3_SHAPES` records the Qwen3-0.6B and 1.7B shapes so configs and parameter counts can be
planned without downloading weights (`python -m chained_transformer.checkpoint plan`); a test
checks the 0.6B shape against the published counts; the generated configs and counts are in
`configs/qwen3-0.6b/` and `configs/qwen3-1.7b/`. `mode_slice(model, m)` returns one mode's
tensors, for single-mode upgrades.

## Single-mode upgrades (`chained_transformer/upgrade.py`)

A mode pack is one mode's tensors (`mode.safetensors`) plus `mode.json`: the mode, the variant,
the parameter count and a trunk fingerprint, a SHA-256 over every parameter that belongs to no
mode. `retrain_mode` freezes everything except one mode (`full` copies or `lora` adapters; `moe`
has no per-mode parameters) and trains it on rows in the dataset format, with loss on that
mode's tokens only (`loss_on="mode"`, the default) or on every token. `swap_mode` copies a pack
into a model after checking variant, mode, trunk fingerprint, tensor names and shapes, so a
pack trained on one trunk cannot be dropped silently into another.

`measure` reports per-label loss and token accuracy and, given a reference model, the KL from
it and the largest log-probability change. Each label is split at the first code position:
before it the code MLP has not run, so a code upgrade must leave those positions bit-identical,
and the split shows whether anything leaks. After it, `think` and `answer` can still move,
because later layers attend to hidden states the new code MLP produced; that drift is what an
upgrade has to keep small.

On a GPU host, with a trained checkpoint and a dataset slice:

```sh
python -m chained_transformer.upgrade retrain --checkpoint CKPT --data DATA --mode code --steps 200 --out PACK
python -m chained_transformer.upgrade swap --checkpoint CKPT --pack PACK --out CKPT-upgraded
python -m chained_transformer.upgrade measure --checkpoint CKPT-upgraded --reference CKPT --data DATA --limit 200 --out measure.json
```

In bf16, on any device, `retrain` uses training's default optimizer (`ChunkedAdamW8bit`, with bf16
stochastic rounding). It writes `train.jsonl` next to the pack. Then run `evals.run` on the upgraded
checkpoint. `tools/upgrade_demo.py` runs the whole idea on a tiny synthetic task on CPU; see
[results/upgrade-demo](../results/upgrade-demo/result.md).

## Generation (`chained_transformer/generate.py`)

`mode_tracking(model, routing, texts)` registers a forward pre-hook. On prefill it starts one
tracker per batch row and feeds it the prompt; on each decode step it feeds the new token and
sets routing. It works with Hugging Face `generate`, one cache throughout, and records the
per-row mode trace. `TokenTexts` decodes each vocabulary id once.

`generate_separate_models` is the `separate` baseline: one model and one cache per mode.
When the mode changes, the incoming model prefills every token it has not yet seen, then
decodes. It reports switches and re-prefilled tokens.

## Data (`chained_transformer/data.py`, `chained_transformer/mix.py`)

Step by step, with every output field: [data.md](data.md).

`python -m chained_transformer.data` samples whole parquet row groups of OpenThoughts3 at random
(weighted by rows, default 400 groups of 12 rows), reads only the rows it needs but must fetch
each chosen row group whole, so it prints the compressed size first and stops adding groups at
`--max-download-gb` (default 6). It renders the Qwen chat format by hand,
tokenizes, labels position modes and profiles every row: domain, length, completeness, tokens
per segment label, mode switches, where code fences fall and where the answer starts. It then
picks the training rows with a mix policy and writes an Arrow dataset with `input_ids` (int32),
`mode_ids` and `loss_mask` (int8), domain, source and completeness flags, plus `stats.json`
(the sample's switching statistics, every candidate policy's token shares, the chosen policy)
and `sample.jsonl` (a few rows shown as mode runs with short snippets). The loss mask covers the
assistant turn including `<|im_end|>`. `--head` keeps the old layout (first rows of evenly
spaced shards).

The dataset is sorted by domain (code shards first, science last), so reading the first rows of
a few shards gives no science and a skewed code/math split; random row groups avoid that.

Mix policies (`mix.CANDIDATES`) keep complete rows only and never truncate: a row longer than
the policy's `max_tokens` is left out rather than cut, because truncation drops most `answer`
tokens. `equal-domains-16k` (the default) gives code, math and science equal shares of loss
tokens; `natural-16k` keeps the sample's own domain mix; `-8k` variants exist for the 11 GB
card. Train at `--seq-len` equal to the policy's `max_tokens` so no row is truncated. The
rebalance is by domain, because within a complete row `think` holds about 95% of loss tokens
whatever the domain; see the [slice result](../results/ot3-slice-composition/result.md).

## Training (`chained_transformer/train.py`)

One loop serves every arm: `--arm`, plus `--mode` for `separate` (loss only on that label) and
`--only-mode` for single-mode retraining. Precision is bf16 weights with an 8-bit AdamW and
bf16 stochastic rounding, which keeps tiny updates that plain bf16 rounding would drop.
Activation checkpointing is on by default. The LM-head loss runs in checkpointed chunks of 256
tokens (`--loss-chunk`), so a full 152k-vocabulary logits tensor is never stored.

Memory choices, all aimed at the 11 GB 2080 Ti (compute capability 7.5):

- `--optimizer chunked` (default) is `optim.ChunkedAdamW8bit`: bitsandbytes-style blockwise
  8-bit moments (blocks of 256, dynamic map), eager, updating each parameter in chunks of
  `--optim-chunk` elements so its fp32 temporaries stay small. torchao's `AdamW8bit`
  (`--optimizer torchao`) computes each parameter's update on the whole tensor; run without
  `torch.compile` its 8-bit quantizer builds int64 code tensors, about 61 bytes of temporaries
  per element of the 155.6M-element embedding (roughly 9 GB), which alone overflows the card.
- `--attention auto` (default) registers `ct_fused_sdpa`: scaled-dot-product attention limited
  to the flash, memory-efficient and cuDNN kernels. Below compute capability 8.0 no fused kernel
  takes bf16, so PyTorch falls back to the math kernel, which stores the full fp32 score matrix
  (about 1 GB per tensor at 4,096 tokens, several during backward). `ct_fused_sdpa` upcasts q, k
  and v to fp32 there instead, which the memory-efficient kernel accepts. If no fused kernel
  fits, it falls back to math with a warning and counts it in `summary.json`
  (`attention_math_fallbacks`).
- `PYTORCH_ALLOC_CONF=expandable_segments:True` is set unless already set, so freed attention
  and loss buffers do not leave the cache fragmented around the gradients allocated amid them.
- `--freeze-embeddings` freezes the tied embedding/LM head (155.6M parameters), saving its
  gradient and optimizer state.
- `--compute-dtype float16` runs the forward and backward under fp16 autocast while the weights,
  gradients and 8-bit AdamW stay bf16, with a dynamic loss scale (starts at 2^14, halves and
  skips the step on a non-finite gradient, doubles after 200 good steps; logged as
  `loss_scale`, skips as `skipped_steps` in `summary.json`). Turing has fp16 but no bf16 tensor
  cores, and its fused attention takes fp16, so this is about 3.3 times faster there at the
  same loss; Qwen3's largest activation (about 8,100) fits fp16. Default `bfloat16`.

`chained_transformer/memory.py` estimates each arm's memory by phase (weights, gradients,
optimizer state, saved activations, layer backward, loss backward, optimizer step) for the
current and the fixed harness: `python -m chained_transformer.memory --harness fixed --seq-len
4096 8192 16384`. Its per-element coefficients are guesses until `tools.memory_check` measures
them (CPU with a tracking dispatch mode, or CUDA with per-phase peak stats). In the `lora` arm the base MLP weights are
frozen and the trunk trains; in `full` and `moe` everything trains. Each step logs total loss,
loss by segment label, MoE balance loss, gradient norm, step time and peak memory to
`train.jsonl`; `summary.json` holds the median step time, tokens/s and peak memory.

Rows longer than `--seq-len` are still truncated (counted as `truncated_rows` in `train.jsonl`);
build data with a policy whose `max_tokens` matches `--seq-len` to avoid that. One row per
micro-batch.
Checkpoint saving is not implemented yet.

## Evals (`evals/`)

`python -m evals.run` loads a mode checkpoint, a plain checkpoint, a Hub name or (for `separate`)
three plain fine-tunes, generates through the mode hook or the separate-models loop, and scores
MATH500 (math-verify on the answer segment), HumanEval+ (evalplus) and LiveCodeBench (a
re-implementation of `lcb_runner`'s grading in a subprocess sandbox, `release_v6`, contest dates
2024-10-01 to 2025-05-01). Every response is split into think, code and answer text by the same
`ModeTracker`, which also drives the thinking-leakage metrics. Each run writes `report.json`
(model, generation settings, git commit, data sources with hashes, scores, leakage, tokens by
segment label, switches, tokens/s, peak memory) and per-benchmark `responses.jsonl` and
`scores.jsonl`; `--score-only` rescores saved responses and `--oracle` scores reference answers
to check a scorer. Details, checks and known deviations from the official harnesses are in
[evals/NOTES.md](../evals/NOTES.md). `chained_transformer/evals.py` is the older stub and can be
deleted.

These use our own generation and scorers rather than lm-eval-harness, because every arm must
generate through the mode hook (or the separate-models loop).

`--decode graph` (greedy only, single model) keeps the refill scheduler, stopping rules and
mode tracking but replaces `generate`'s decode loop: after a chunked prefill into the
preallocated cache, each step runs a hand-written Qwen3 decode
(`chained_transformer/graph_decode.py`) captured once per 512-token cache-length bucket as a
CUDA graph and replayed. Attention is the fused memory-efficient kernel with grouped queries
over the cache up to the bucket; mode MLPs run every mode on the batch's rows and keep each
row's own (`moe` runs every expert and mixes the chosen ones as `MoEMLP.mix` does). On CPU it
reproduces `generate` token for token; on the GPU it differs from `generate` only by kernel
rounding. Use it with `--dtype float16` on Turing, where bf16 matmuls are emulated.

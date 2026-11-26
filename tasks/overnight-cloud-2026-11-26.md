# Overnight cloud work, 2026-11-26: harness memory, MoE drift, data mix

Branch `cloud/harness` off `main` at 63f337b, built in a CPU-only cloud container (4 cores,
15 GB RAM) while Philip slept. Delivered as a git bundle and patches in the project folder,
`chained-transformer/cloud-harness/`, because the container cannot push.

## Blockers

1. **Hugging Face is blocked from the container.** huggingface.co and the datasets server
   return 403 through the container's proxy. Not routed around. So no Qwen3-0.6B weights, no
   tokenizer and no OpenThoughts3 here: real-model checks, the data build and the probes are
   scripts for the 5080.
2. **No torch in the container.** Installing the packages `pyproject.toml` lists needed
   Philip's yes, asked at 09:31 UTC, and none came before he slept. Nothing was installed. So
   no torch code on this branch has run: the only tests run are the pure-Python mix tests
   (`tests/test_mix.py`, 5 passed) and ruff (clean). Every torch test added here is unrun.

What stood in for running code: reading the pinned dependencies' own source (torch 2.14.0,
transformers 5.24.1, torchao 0.18.0, unpacked from PyPI wheels and read, never installed or
run) and a numpy emulation of bf16 rounding.

## 1. Out of memory on the 2080 Ti

Every arm, even `sft`, ran out of memory at 4,096 tokens on the 11 GB card. Causes, from the
source, in order of size:

1. **torchao's 8-bit AdamW ran uncompiled.** torchao calls `torch.compile` on its per-parameter
   step, but torch 2.14's Inductor skips any bf16 graph on a GPU below compute capability 8.0
   (`_check_triton_bf16_support` in `torch/_inductor/compile_fx.py` raises
   `SkipFrame("BF16 is not supported")`, and Dynamo then runs the frame eagerly even with
   `fullgraph=True`). Uncompiled, `single_param_adam` holds four fp32 copies of the parameter
   for the whole step, and `quantize_8bit_with_qmap` builds its codes as int64 tensors: about
   61 bytes of temporaries per element at the peak. For the tied 155.6M-element embedding that
   is roughly 9 GB on top of weights, gradients and state (3.4 GB for `sft`). This alone
   exceeds the card. On the 5080 (compute capability 12.0) the step compiles, so the 5080 never
   showed this.
2. **Attention ran in the math kernel in fp32.** Below compute capability 8.0 none of PyTorch's
   fused attention kernels takes bf16, so SDPA uses the math kernel, which upcasts to fp32 and
   stores the full score matrix: about 1 GB per tensor at 4,096 tokens and 16 heads, around 3 GB
   during each layer's recomputed backward, growing with the square of the length.
3. **Loss chunks of 1,024 tokens** cost about 2 GB of transient fp32 logits and gradients.
4. **Fragmentation**: gradients allocated in the middle of large freed attention and loss
   buffers keep those segments reserved.

Fixes on this branch (each is a flag, so the old behaviour stays reachable):

- `--optimizer chunked` (default): `ChunkedAdamW8bit`, the same blockwise 8-bit state as
  bitsandbytes and torchao (blocks of 256, dynamic map, bf16 stochastic rounding), eager, applied
  in chunks of 4M elements so its temporaries stay around 160 MB whatever the parameter size.
  Never compiled, so it behaves the same on both cards.
- `--attention auto` (default): a registered attention function that allows only the fused
  kernels and, below compute capability 8.0, upcasts q, k and v to fp32 so the memory-efficient
  kernel accepts them. Checked line by line against transformers 5.24.1's own
  `sdpa_attention_forward` (causal rule, key slicing, mask registration). If no fused kernel
  fits it warns, falls back to math, and counts it in `summary.json`.
- `--loss-chunk 256` (default, was 1,024).
- `expandable_segments:True` for the CUDA allocator unless already set.
- `--freeze-embeddings` (off by default), which saves about 0.6 GB.

Estimated peak GiB per arm (`python -m chained_transformer.memory`; CUDA context not included,
coefficients not yet calibrated, so treat these as estimates):

| arm | current, 2080, 4k | fixed 4k | fixed 8k | fixed 16k | fixed 16k, frozen embeddings |
| --- | ---: | ---: | ---: | ---: | ---: |
| sft | 12.2 | 4.2 | 5.1 | 6.9 | 6.3 |
| full | 15.2 | 7.2 | 8.2 | 10.0 | 9.4 |
| lora | 11.6 | 3.7 | 4.6 | 6.5 | 5.9 |
| moe | 15.2 | 7.4 | 8.5 | 10.8 | 10.2 |
| separate | 12.2 | 4.2 | 5.1 | 6.9 | 6.3 |

Reading: every arm should fit the 11 GB card at 4k and 8k with at least 2 GB to spare; at 16k,
`full` and `moe` have no margin on that card and belong on the 5080. The current-harness
column matches what the 2080 smoke saw (every arm out of memory at 4k).

To measure instead of estimate, on the 2080 Ti:

```sh
uv run --locked --no-sync python -m tools.memory_check --device cuda --base Qwen/Qwen3-0.6B --seq-len 4096 8192 16384 --budget-gib 10.5
```

It runs each arm for 3 steps in its own process, records peak allocated and reserved GiB per
phase (setup, forward, backward, optimizer) and the phase that ran out of memory, compares
with the estimate and exits 1 if anything overflows. `--harness current-turing` reproduces the
old setup for a before/after. The CPU lane (`--device cpu --layers 2`) counts live tensor bytes
with a dispatch-mode tracker and is for calibrating the estimate's coefficients.

## 2. MoE drift in bf16

Cause: with identical experts, `MoEMLP` cast the renormalised routing weights to bf16, rounded
each weighted expert output to bf16 and summed them in a bf16 buffer, so the two weights did
not add back to exactly one and each sum rounded again. Fix: route in fp32, multiply and
accumulate in an fp32 buffer, cast once at the end.

Evidence (numpy emulation of both code paths with bf16 rounding, 4,096 tokens x 1,024 wide,
top-2 of 3 identical experts; script in the delivery folder, `emulation/moe_bf16_combine.py`):
the old combine changed 18.6% of the MLP output's elements, by up to 0.78% (mean 0.095%); the
fixed combine returns the base output exactly. Compounded over 28 layers, that size of error
is consistent with the 0.4 to 0.66 logit gap the 2080 smoke saw (inferred, not measured).

Tests added (unrun): identical experts in bf16 return exactly the expected output, and the
bf16 `moe` arm's logits stay as close to the base model's as the `full` arm's.

## 3. Data rebuild

`python -m chained_transformer.data` now samples 400 random parquet row groups (12 rows each)
instead of the first rows of a few shards, profiles every row (switches per trace, tokens per
mode, where code fences fall, where the answer starts, length) and selects training rows with
a mix policy. It writes the dataset, `stats.json` (switching statistics, each candidate
policy's token shares) and `sample.jsonl`. Not run: it needs Hugging Face.

Length policy, proposed: complete rows only, never truncate, and leave out rows longer than
the training length. Default `equal-domains-16k` (code, math and science get equal loss-token
shares, rows up to 16,384 tokens); `equal-domains-8k` for the 2080 Ti. Rebalancing by mode
inside a row is not possible (a complete row's loss tokens are about 95% `think`), so the
rebalance is across domains, which moves the code share. `train.py` still truncates rows
longer than `--seq-len` and now counts them, so pair the policy's length with `--seq-len`.

On the 5080:

```sh
uv run --locked --no-sync python -m chained_transformer.data --policy equal-domains-16k
```

## 4. Probes and design note

- `tools/mode_probe.py`: logistic probes on base Qwen3-0.6B hidden states per layer, for
  position mode and for segment label, balanced accuracy on held-out rows against a majority
  baseline. Unrun. `uv run --locked --no-sync python -m tools.mode_probe --data artifacts/data/ot3-mix`.
- `docs/routing-and-looping.md`: how mode routing could combine with looped layers (Relaxed
  Recursive Transformers, Mixture-of-Recursions, MoEUT checked tonight), the KV-cache question
  it raises, and a first experiment with a kill rule.

## Measured versus estimated

- Measured: nothing on torch. The mix logic (5 tests) and the bf16 combine (numpy emulation).
- Read from source: torch 2.14 skips compiling bf16 below sm80; torchao 0.18's uncompiled step
  allocations; transformers 5.24.1's attention interface.
- Estimated: every memory number above.

## Order on the GPU

1. `python -m tools.check` on either machine: the new torch tests run for the first time.
2. The memory check on the 2080 Ti (command above).
3. `python -m tools.gpu_smoke` on the 5080 when it is free, then the data build and probes.

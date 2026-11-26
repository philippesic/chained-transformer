# First experiment: five arms on Qwen3-0.6B

## Goal

Find out whether hard-routed full MLP modes (`full`) beat a plain fine-tune (`sft`) and a
same-size per-token MoE (`moe`) at equal training tokens, and whether one mode can be retrained
alone without hurting the others. Arms, metrics and decision rules are in the
[README](../README.md#first-experiment).

## Authorized

- 2026-11-25 (Philip's setup brief): repo setup, model surgery, generation hook, data stub,
  eval stubs, smoke tests on the 5080; downloading Qwen3-0.6B and an OpenThoughts3 slice. No
  long training runs yet. Local repo only: no GitHub repo, no push. Do not touch BitDraft's
  checkout or stop its runs; skip GPU work while another job uses the GPU. No system-level
  installs. Sub-agents must not use fast mode.
- Kill rule (Philip, 2026-11-25): drop the idea if `full` does not beat `sft` and `moe` by
  about 2 points across two seeds. If `lora` ties `full`, continue with LoRA modes.

## Done when

- GPU smoke passes: identical modes reproduce base generation on the GPU, all arms fit in
  16 GB at the chosen sequence length, and a 50-step run does not diverge; peak memory and
  step time are recorded here.
- The open decisions below are settled with Philip and recorded with dates.
- Five arms are trained on the same tokens with two seeds, evaluated, and the result is
  written up with close-results.

## State

Owner: the setup thread (Claude), branch `main`, no remote.

Done and verified:

- Environment: uv, CPython 3.11.15, torch 2.14.0+cu130 (sees the RTX 5080), transformers
  5.24.1, torchao 0.18.0; `uv.lock` committed.
- `python -m tools.check` passes (45 CPU tests: tracker edge cases, surgery equivalence and
  routing, one cache across switches, single-mode freezing, MoE router gradients, generation
  hook, the separate-models baseline, data labelling, eval helpers, every arm training 4 steps
  on a tiny model).
- Real Qwen3-0.6B on CPU, fp32, 160 new tokens, 3 prompts: `full`, `lora` and `moe` with
  identical modes produce exactly the base model's greedy tokens. Two prompts switched
  answer → code → answer on one cache; the thinking prompt stayed in `think` within 160
  tokens. Max logit difference about 5e-5 (row-gathered matmuls in a mixed-mode prefill).
- Parameter counts: base 596.0M (MLP 44.3%), `full` 1,124.5M, `lora` 662.1M (rank 64),
  `moe` 1,124.6M.
- Data slice `artifacts/data/ot3-slice-4k` (4,000 rows, 62.7M tokens, 359 MB, built in 90 s).
  See the [slice result](../results/ot3-slice-composition/result.md): only 27.6% of rows are
  complete, and `code`/`answer` get about 1%/4% of loss tokens.

Blocked: the GPU smoke (`python -m tools.gpu_smoke`) has not run. On 2026-11-26 at about
00:00, BitDraft's 4-bit control run (`control-run-3`, pueue `gpu` group) held 12.2 GB of the
GPU at 99% utilisation. No GPU memory or step-time numbers exist yet.

Open decisions (proposals, not yet agreed):

1. **Training data.** Proposal: train on complete rows only, and decide whether to rebalance
   toward code- and answer-heavy rows, since equal tokens otherwise starve those modes. The
   slice also has no science rows.
2. **Sequence length.** Complete rows run up to 16k tokens. Proposal: decide after the GPU
   smoke's 16k memory probe; the alternative is truncating, which drops most answer tokens.
3. **Prompt tokens' mode.** Currently `think` (`prompt_mode`). An alternative is `answer`.
4. **`lora` arm trainables.** Currently: base MLP frozen, trunk and LoRA trainable.
5. **`moe` top-k.** Currently 2 (as in BTX); top-1 would match `full`'s active compute.
6. **LiveCodeBench.** Needs a pinned `lcb_runner` commit; not wired yet.

Missing before real runs: checkpoint save and load, row packing or a length policy, and an eval
CLI that loads trained arms (the current CLI runs the base model only).

## Next

Philip decided on 2026-11-26 not to queue the GPU smoke behind BitDraft: he will say when the
GPU is free. Then run `python -m tools.gpu_smoke`, record memory and step time here, and settle
decisions 1 and 2.

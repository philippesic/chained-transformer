# First experiment: five arms on Qwen3-0.6B

## Goal

Find out whether hard-routed full MLP modes (`full`) beat a plain fine-tune (`sft`) and a
same-size per-token MoE (`moe`) at equal training tokens, and whether one mode can be retrained
alone without hurting the others. Arms, metrics and decision rules are in the
[README](../README.md#first-experiment).

## Authorized

- 2026-11-25 (setup brief): repo setup, surgery, generation hook, data, eval stubs, smoke tests.
  Do not touch BitDraft's checkout or runs. Sub-agents must not use fast mode.
- Kill rule (2026-11-25): drop the idea if `full` does not beat `sft` and `moe` by about
  2 points across two seeds. If `lora` ties `full`, continue with LoRA modes.
- 2026-11-26: the pilot `pilot-0p6b-8k` on the 2080 Ti (pueue group `ct`), kill rule
  `seed-mean`, seeds 0 and 1, the verdict eval set, `--seq-len 8192`, data `ot3-natural-8k`.
- 2026-11-28 about 01:15: the overnight go ("do this as well as general research and
  experimentation"), the 2080 Ti once the ShaderLOD promo thread finished (it did at 01:31).

## Done when

- The open decisions below are settled with Philip and recorded with dates.
- Four verdict arms are trained on the same tokens with two seeds, evaluated, and the result is
  written up with close-results.

## State

Owner: the overnight thread "Progress Recap and Overnight Options" (2026-11-28). Branches:
`cloud/moe-fix` (the pilot's code, main checkout), `overnight/fp16` (worktree
`~/ct-overnight`, the speed work below). Both pilots below run code from a checkout that must
not change while they run.

### Pilots

`pilot-0p6b-8k` (bf16, the approved plan, `artifacts/pilot/pilot-0p6b-8k`): `sft-s0` trained
(1,000 steps); its eval has 5 of 888 answers, kept for `--resume`; every other run pending.
Paused 2026-11-28 02:25 for the candidate below. Resume:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --locked --no-sync python -m
tools.pilot run --root artifacts/pilot/pilot-0p6b-8k` (through jobwatch, group `ct`, from the
main checkout).

`pilot-0p6b-8k-fp16` (candidate, `artifacts/pilot/pilot-0p6b-8k-fp16`, runs from worktree
`~/ct-candidate` pinned at `overnight/fp16` 89070bd): the same plan with
`--train-extra="--compute-dtype float16"` and `--eval-extra="--attention lean --dtype float16
--decode graph"`; nothing else differs (diffed). `sft-s0` training started 02:27 (jobwatch
`ct-cand-train-sft-s0`), its eval chained (`ct-cand-eval-sft-s0`). If Philip adopts the
settings, the pilot continues in this folder; if not, it is evidence and can be deleted.

### Speed findings, 2026-11-28 (2080 Ti, measured)

- The 2080 Ti (Turing) has no bf16 tensor cores: 8192x1024x3072 matmuls run at 4.2 TFLOPS in
  bf16, 7.2 in fp32 and 27.5 in fp16 with fp32 accumulation (measured beside a running eval).
  Fused attention has no bf16 kernel on sm_75, hence the existing fp32 copies.
- Qwen3 activations fit fp16: on three 8k rows the largest value at any linear layer or in the
  residual stream of `sft-s0` is about 8,100 (layer 2's massive activation), against fp16's
  65,504.
- Training in fp16 (autocast, bf16 weights and 8-bit AdamW unchanged, dynamic loss scaling):
  `sft` at 8k, 8 rows per step, 15.0 s per step against 49.8 in bf16 (3.3x), loss 1.3315 vs
  1.3319 at step 0 and 1.2055 vs 1.2059 at step 3 on the same rows. `full` 13.1 s and `moe`
  15.4 s per step, peaks 8.1 and 8.4 GiB (bf16 `full`: 8.0). Projected training for the 8 runs:
  about 33 h instead of about 111 h.
- Eval decode in bf16 is GPU-bound in emulated bf16 matmuls (62% of a 39.5 ms step at batch 5,
  2k context) and full-cache fp32 copies (20%), with about 36 ms of Python per step behind it.
  So fp16 alone gains 9%. `--decode graph` (new) replays each greedy decode step as a CUDA
  graph over a preallocated cache, with fused fp16 attention over 512-token length buckets.
  15 MATH500 problems, 2,048-token cap, batch 5: 68 s against 196 s (fp16, current decode)
  and 237 s (bf16, current path); `full` 86 s, `moe` 96 s.
- Agreement: on CPU, graph decode reproduces `generate` token for token for all four arms
  (tests); on the GPU, a captured graph equals the uncaptured step. Against the current decode
  at the same fp16 precision, answers share a median of about 2,300 characters before kernel
  rounding makes them diverge (3 of 15 identical); fp16 against bf16 diverges after about 300.
- The bf16 eval runs slower than recorded: the step-time model with the base model's answer
  lengths gives about 17 h per arm eval at batch 5, and the first 8 minutes ran at 37 answers
  per hour (about 24 h). Eight evals would take 140 to 190 GPU-hours on that path.

### Earlier results

See [FINDINGS](../FINDINGS.md): slice composition, the toy upgrade demo, `sft-s0` loss against
base (0.157 nats per token below on unseen rows), the CPU eval rehearsal, eval decode cost, and
the mode probe (80% balanced accuracy at layer 2, about 90% from layer 12).

### Decisions (Philip)

- 2026-11-26: pilot on the 2080 Ti from `cloud/moe-fix`, `seed-mean`, seeds 0 and 1, verdict
  eval set, 8k, data `natural-8k` (2,246 complete rows, about 3.6 passes per run).
- 2026-11-27: evals use `--attention lean` with `--eval-batch-size 5`; LiveCodeBench release_v6
  through April 2025, 341 problems (the plan's 361 selects all 341).
- 2026-11-27 11:35: pilot stopped (`full-s0` at step 44, discarded).

### Open decisions

1. **Adopt fp16 training and graph decode** for every remaining run (the candidate pilot)?
   Costs: `sft-s0` retrains (about 4 h, running now); answers differ from bf16 the way any
   precision change does. Gain: the pilot drops from roughly 11 to 13 GPU-days to roughly 3.
2. **Prompt tokens' mode**: currently `think`; the alternative is `answer`.
3. **`lora` trainables**: base MLP frozen, trunk and LoRA trainable.
4. **`moe` top-k**: 2 (as in BTX); top-1 would need the router probability as the weight.

The data question is settled for the pilot (complete rows, natural mix), but `code` gets 0.39% of
loss tokens, so the `full` arm's code MLPs learn from about 2,500 tokens per step on the 60% of
steps that have any. Adam's per-weight scaling means they still move at about the learning
rate on those steps; measure per-mode weight movement once `full-s0` exists.

## Next

- Candidate `sft-s0`: compare its 1,000-step loss curve with the bf16 run's (same rows, same
  order); then its eval, about 5 h by the 3.5x measurement.
- Morning: Philip decides 1. If adopted, continue `tools.pilot run` on the candidate root from a
  checkout of the merged code; if not, resume the bf16 pilot and delete the candidate folder.
- Disposal: `~/ct-overnight/artifacts/{fp16-check,graph-check}` (about 5 GB, two converted base
  checkpoints) once the speed result is written; `sft-s0/eval-auto-oom-20261127` once the report
  is written.

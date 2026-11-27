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

GPU smoke on the 2080 Ti (2026-11-26, raw logs on that host in
`artifacts/smoke/2080-20261126-004755/`, disposable once the 5080 smoke runs): bf16 generation
with identical modes matched the base tokens exactly for `full` and `lora`; `moe` diverged
(max logit difference 0.4 to 0.66), most likely because expert outputs are weighted and summed
in bf16. fp32 `full` matched exactly. Every training run, even `sft`, ran out of memory on the
11 GB card at 4,096 tokens, failing in torchao's stochastic-rounding step; why `sft` needs over
10 GB is not yet understood. The investigation stopped there.

Blocked on the 5080: the GPU smoke (`python -m tools.gpu_smoke`) has not run. On 2026-11-26 at about
00:00, BitDraft's 4-bit control run (`control-run-3`, pueue `gpu` group) held 12.2 GB of the
GPU at 99% utilisation. No GPU memory or step-time numbers exist yet.

Decided 2026-11-26 (Philip, via the Duet thread): run the pilot on the 2080 Ti from branch
`cloud/moe-fix` with kill rule `seed-mean`, seeds 0 and 1, the verdict eval set and
`--seq-len 8192`. The proposed `equal-domains-8k` mix held only 55 complete rows (each would be
seen about 145 times in 1,000 steps of 8 rows), so Philip chose a larger `natural-8k` build
instead: `artifacts/data/ot3-natural-8k`, 2,246 complete rows (science 1,520, math 503,
code 223; 11.6M tokens, mean 5,156 per row), about 3.6 passes in a training run. This settles
decisions 1 and 2 for the pilot. `plan --replace` can still change settings before any eval runs.

Pilot `pilot-0p6b-8k` (pueue `ct` task 1691, started 2026-11-26 about 21:00 local):

- 2080 Ti verification on `cloud/moe-fix`: `moe` matches base 3/3 on both smoke benchmarks
  and is bitwise exact in `gpu_equivalence` at batch 1 and 3; memory at 8k peaks at 8.3 GiB.
- Measured 8k step times per row: sft 11.2 s, full 11.8 s, lora 11.5 s, moe 14.1 s
  (`artifacts/pilot-budget/2080-8192.json`). Estimate to the verdict: 6.9 to 10.4 days
  (training about 111 h measured, evals 55 to 138 h assumed). Kept at 8k because every row
  fits and 4096 cuts most answers.
- Preflight: the question is whether `full` beats `sft` and `moe` by about 2 points. The plan
  assumed 361 LiveCodeBench problems, but the eval window (2024-10-01 to 2025-02-01) holds
  only 166 (found 2026-11-27 by the overnight thread). After `sft-s0`'s eval the sample grows
  math500 to all 500 and that eval resumes for the extra problems. Under `seed-mean` with
  2 seeds, a true 2-point gap then continues 59.5% of the time at 10% discordance (42% at 15%,
  31% at 20%), down from 69%; no gap stays at 1%. Widening LiveCodeBench beyond the window
  (all 880 of release_v5: 79%) is Philip's call; `--resume` keeps answered problems. Kill or repair on a training traceback, NaN or diverging loss,
  an eval out-of-memory, or step times over 1.5 times the measured ones. The pilot stops
  cleanly (exit 3) when another process holds the GPU at a step boundary; restart it with the
  same `pilot run` command once the card is free.
- Eval memory: commit eb7987d (from the overnight thread, CPU-tested only) keeps refilled eval
  batches within the length their batch size was budgeted for and prefills in 512-token
  chunks; without it a carried 8k row was expected to run out of memory. If the first eval
  still runs out of memory, `plan --replace` with `--eval-extra="--no-refill"`; answers are kept.
- Disk: about 160 GB free; 8 checkpoints need about 14 GB. Disposal: keep the checkpoints
  until the report is written, then keep only what close-results names.
- Progress: `uv run --locked --no-sync python -m tools.pilot status --root
  artifacts/pilot/pilot-0p6b-8k`.

Overnight thread 2026-11-27 (CPU only, nothing pushed; branches sit on `cloud/moe-fix`):

- `overnight/speed-on-moe-fix` (d06f389): `cloud/moe-fix` at 76cb559 plus `tools/loss_vs_base.py`,
  the eval speed work (`--attention lean`, length-grouped refill, preallocated cache) and three
  result pages (`results/sft-loss-vs-base`, `results/first-eval-rehearsal`,
  `results/eval-decode-cost`). Forecast about 28 h per arm eval on the current path, about 9 h
  with the branch; the GPU slope is from two points only.
- Commit 47c227e (on top of that branch): LiveCodeBench window of 341 problems (release_v6
  through April 2025), continue odds 71% at a true 2-point gap instead of 59.5%.
- Findings: sft-s0 is 0.157 nats per token below base on fresh rows by the end of pass 1
  (two thirds of it within 20 steps); all four verdict arms rehearsed end to end on CPU;
  about 30% of base MATH500 answers hit the 8,192 cap and hold most decode tokens.

- Mode probe (runbook step 8, CPU, 55 full-length rows of `ot3-mix-8k`, 3 classes, 33%
  baseline; `results/mode-probe/result.md` on the overnight branch): held-out balanced accuracy
  for position mode by layer: 0: 0.545, 1: 0.728, 2: 0.800, 4: 0.862, 12: 0.895, 20: 0.908,
  28: 0.899; segment label within 0.01. Use `--max-tokens 8192`; the 4,096 default keeps 12
  rows and no code positions.

2026-11-27 10:52: `sft-s0` trained (1,000 steps, checkpoint saved), but its first eval ran out of
GPU memory 15 minutes in, inside `fused_sdpa`'s fp32 copies of the key and value cache, after
4 MATH500 answers (all finished, all closed thinking). The auto batch (7 rows) budgets the KV
cache but not those per-layer fp32 copies. While Philip decides the eval settings, `full-s0` is
training via `pilot train-one` (pueue `ct` task 1695, jobwatch `ct-pilot-train-full-s0`), so
no GPU time is spent on an eval path that may change. Restart with the usual `pilot run`
command once the eval settings are in the plan.

GPU decode bench, 2080 Ti, base model, bf16 (`artifacts/bench/2080-fit.json`): ms per decode
step, current `auto`+`dynamic` against `lean`+`roomy`: 7x1024 55.8 vs 37.5, 7x4096 130.9 vs
61.5 (2.1x), 16x2048 144.3 vs 66.3 (2.2x); `same_argmax_rate` 0.994 to 0.995 (bf16 rounding,
max logit difference 0.6 to 0.7). At 7x8192 both paths ran out of memory in the bench's
single-pass prefill; the eval prefills in 512-token chunks. A fixed `--eval-batch-size 5`
leaves about 2 GB of headroom even for `full` and `moe`.

Open pilot decisions for Philip (record his answers here with the date):

1. Speed branch: pause at the end of `sft-s0` training with the pilot's own GPU-busy stop (a
   small CUDA holder, never killing the driver), run `tools.decode_bench` (about 4 minutes),
   and adopt only if `lean`+`roomy` is at least 1.5 times faster than `auto`+`dynamic` at
   7x4096 and 7x8192 with `same_argmax_rate` at least 0.99. All arms must share the choice, so
   it must happen before `full-s0`'s eval. Without a yes, read the eval's tokens per second
   one hour in (about 11:35) and report.
2. LiveCodeBench window: keep 166 problems or take 47c227e's 341 before more evals run.

Open decisions (proposals, not yet agreed):

1. **Training data.** Proposal: train on complete rows only, and decide whether to rebalance
   toward code- and answer-heavy rows, since equal tokens otherwise starve those modes. The
   slice also has no science rows.
2. **Sequence length.** Complete rows run up to 16k tokens. Proposal: decide after the GPU
   smoke's 16k memory probe; the alternative is truncating, which drops most answer tokens.
3. **Prompt tokens' mode.** Currently `think` (`prompt_mode`). An alternative is `answer`.
4. **`lora` arm trainables.** Currently: base MLP frozen, trunk and LoRA trainable.
5. **`moe` top-k.** Currently 2 (as in BTX); top-1 would match `full`'s active compute. With
   top-1 the renormalised weight is always exactly 1, so the router would learn only from the
   balance loss; top-1 needs the router probability itself as the weight.
6. **LiveCodeBench.** Needs a pinned `lcb_runner` commit; not wired yet.

Missing before real runs: checkpoint save and load, row packing or a length policy, and an eval
CLI that loads trained arms (the current CLI runs the base model only).

Branch `cloud/evals` (2026-11-26 overnight, CPU only, Hugging Face blocked) adds mode
checkpoints with save and load, an eval CLI for trained arms (MATH500, HumanEval+,
LiveCodeBench, leakage), a single-mode upgrade harness and a pilot launcher for the five arms;
see `evals/NOTES.md`. Its first GPU step is `python -m tools.real_model_smoke`.

## Next

Branch `cloud/harness` (2026-11-26 overnight, unrun on torch) targets the 2080 Ti out-of-memory
failures, the bf16 `moe` drift and the data mix; see
[overnight-cloud-2026-11-26](overnight-cloud-2026-11-26.md). Run it with
[runbook-cloud-harness](runbook-cloud-harness.md): the check and the 2080 memory check first.

Philip decided on 2026-11-26 not to queue the GPU smoke behind BitDraft and will say when the
GPU is free. Then run `python -m tools.gpu_smoke`, record memory and step time here, and settle
decisions 1 and 2.

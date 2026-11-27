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
- Preflight: the question is whether `full` beats `sft` and `moe` by about 2 points. Under
  `seed-mean` with 2 seeds, a true 2-point gap continues 69% of the time and no gap 1%
  (`tools.eval_power --odds`). Kill or repair on a training traceback, NaN or diverging loss,
  an eval out-of-memory, or step times over 1.5 times the measured ones. The pilot stops
  cleanly (exit 3) when another process holds the GPU at a step boundary; restart it with the
  same `pilot run` command once the card is free.
- Disk: about 160 GB free; 8 checkpoints need about 14 GB. Disposal: keep the checkpoints
  until the report is written, then keep only what close-results names.
- Progress: `uv run --locked --no-sync python -m tools.pilot status --root
  artifacts/pilot/pilot-0p6b-8k`.

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

# First-eval rehearsal on CPU: the path runs, but the eval set and its memory budget were off

**Conclusion.** The pilot's own steps run end to end on the real Qwen3-0.6B for all four
verdict arms (`sft`, `full`, `lora`, `moe`): train, save, eval on the three benchmarks, then
`status`, `compare` and `pilot_report`. LiveCodeBench had not run in any earlier check. The
rehearsal found two defects before the pilot's first eval and cleared three ways the eval
could have been silently wrong.

1. **LiveCodeBench holds 166 problems in the eval window, not 361.** The verdict sample was
   sized from the release (880 problems); the window 2024-10-01 to 2025-02-01 keeps 166. With
   the sample that can actually run (MATH500 500, HumanEval+ 164, LiveCodeBench 166), a true
   2-point gap continues 59.5% of the time under the seed-mean rule, not the 69% the preflight
   stated.
2. **The batch refill outgrew the memory its batch size was budgeted for.** A refilled batch is
   padded to its longest carried row, and new rows then generate up to their own cap, so
   sequences grew past prompt plus max-new-tokens. A test shows it on the old code. Fixed in
   31b0bd8 (each call stops at the budgeted width and prefills in 512-token chunks), which the
   pilot's checkout took as eb7987d before its first eval.

Odds that the pilot continues, by true gap and eval set (`tools.eval_power`, two seeds,
seed-mean rule, arms assumed to disagree on 10% of problems):

| Eval set (MATH500 / HumanEval+ / LiveCodeBench) | Per-seed SE | No gap | 2 points | 3 points |
| --- | ---: | ---: | ---: | ---: |
| As planned, 361 / 164 / 361 | 1.14 | 1.2% | 68.7% | 96.6% |
| What exists, 500 / 164 / 166 | 1.25 | 1.2% | 59.5% | 92.8% |
| `release_v6` window to 2025-05-01, 500 / 164 / 341 | 1.11 | 1.2% | 71.3% | 97.3% |
| Whole `release_v5`, 500 / 164 / 880 | 1.01 | 1.2% | 79.4% | 99.0% |

At 15% disagreement the second row falls to 42%, at 20% to 31%, and nothing is left to grow.

Checks that came back clean:

- The data's `mode_ids` equal the generation hook's routing on 300 training rows (1.51M
  tokens, no mismatch), so the routed arms are evaluated with the routing they trained with.
- Left padding in a batch does not change the modes the hook assigns (18 padded cases).
- The LiveCodeBench scorer passes 6 of 6 hand-written correct solutions on the real hidden tests
  (stdin and functional) and fails 6 of 6 wrong ones (wrong answer, time limit, exception); it
  ignores code inside unclosed thinking.

**Conditions.** CPU only, in a worktree off `cloud/moe-fix` at 2a66137, while the pilot trained
on the GPU. Rehearsal plan: `tools.pilot` with seed 0, 2 training steps of 2 rows at 1,024
tokens, then every benchmark limited to 6 problems, 768 new tokens, batch 3 (so the refill
path runs), fp32. `sft` was evaluated on the old refill and the other three arms on the fixed
one. The `full` checkpoint's 18 answers were also regenerated with the speed branch's decode
path and are identical in text and mode runs. Scores at 6 problems and 768 tokens mean
nothing; the point is that every step ran. Evidence: [rehearsal.json](rehearsal.json).

**Limitations.** Nothing ran on a GPU. The memory claim rests on the width test and on
arithmetic (at batch 7 and 8,000 tokens the padding mask and its fp32 copy alone take about
2.5 GiB next to 6.3 GiB of cache on a card with about 9.2 GiB free); no out-of-memory error
was observed. Two training steps do not test whether an arm trains well, only that it trains,
saves and loads. The 10% disagreement between arms is an assumption until two arms have been
evaluated.

**Disposition.** The refill fix is retained and in the pilot's checkout. Widening LiveCodeBench
is Philip's decision: branch `overnight/lcb-v6` (47c227e) switches to `release_v6` with 341
problems and corrects `eval_power.FULL_SIZES`; answers already generated are kept by
`--resume`. The rehearsal's outputs (`artifacts/pilot/rehearsal` in the overnight worktree,
6.6 GB, mostly its four checkpoints) are disposable; the checkpoints were deleted after this
page was written.

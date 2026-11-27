# Findings

- [OpenThoughts3 slice composition](results/ot3-slice-composition/result.md) (2026-11-26):
  66% of traces lack `</think>`; even complete rows give `code` about 1% and `answer` about 4%
  of loss tokens.
- [Single-mode upgrade on a toy task](results/upgrade-demo/result.md) (2026-11-26, CPU toy):
  retraining only the `code` MLPs keeps thinking bit-identical, but with loss on code alone the
  frozen `answer` mode stops matching the new code; loss on all tokens recovers most of it.
- [Pilot `sft-s0` loss against the base](results/sft-loss-vs-base/result.md) (2026-11-27, CPU
  recomputation of a 2080 Ti run): on unseen rows the fine-tune is 0.157 nats per token below the
  base by the end of pass 1, two thirds of it gained in the first 20 steps; `code` tokens are
  0.39% of the loss and barely move.
- [First-eval rehearsal on CPU](results/first-eval-rehearsal/result.md) (2026-11-27, CPU only):
  train, save and eval run end to end for all four verdict arms; LiveCodeBench's window holds 166
  problems, not the planned 361, so a true 2-point gap continues 59.5% of the time, not 69%; the
  batch refill outgrew its memory budget and is fixed.

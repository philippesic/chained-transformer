# Findings

- [OpenThoughts3 slice composition](results/ot3-slice-composition/result.md) (2026-11-26):
  66% of traces lack `</think>`; even complete rows give `code` about 1% and `answer` about 4%
  of loss tokens.
- [Single-mode upgrade on a toy task](results/upgrade-demo/result.md) (2026-11-26, CPU toy):
  retraining only the `code` MLPs keeps thinking bit-identical, but with loss on code alone the
  frozen `answer` mode stops matching the new code; loss on all tokens recovers most of it.

# OpenThoughts3 slice: mostly truncated thinking, little code or answer

**Conclusion.** In a 4,000-row slice of OpenThoughts3-1.2M, 66% of traces never reach
`</think>`, and only 27.6% are complete (they have `</think>` and fewer than 16,000 assistant
tokens). Even in complete rows, `think` labels 95.3% of loss tokens, `answer` 3.7% and `code`
0.95%. Training on these tokens as they are would give the `code` and `answer` modes about 1%
and 4% of the gradient signal.

| Rows | Rows | think tokens | answer tokens | code tokens |
| --- | ---: | ---: | ---: | ---: |
| All | 4,000 | 60.91M (98.7%) | 0.63M (1.0%) | 0.17M (0.3%) |
| Complete | 1,102 | 13.16M (95.3%) | 0.52M (3.7%) | 0.13M (0.95%) |

Complete rows by domain: math 795 of 3,000 (26.5%), code 307 of 1,000 (30.7%). Median row
length is 16,801 tokens (prompt plus assistant); assistant turns pile up between about 16.1k
and 16.5k tokens, the sign of a generation cap. Truncated traces often end mid-sentence and
some drift into mixed Chinese and English.

**Conditions.** Rows 0 to 249 of 16 evenly spaced shards (`train-00000` to `train-00105`,
step 7), labelled with the tracker in `mode_mlp/modes.py` as of this commit and tokenized with
the Qwen3-0.6B tokenizer. The dataset is sorted by domain: these shards held math and code,
no science. The 16,000-token completeness threshold is a heuristic chosen from that pile-up.

**Limitations.** First rows of each shard only; other positions may differ in difficulty and
length. "Code" counts only fenced blocks; code written in unfenced prose stays `think` or
`answer`.

**Disposition.** Feeds the open data decision in the
[first-experiment task](../../tasks/first-experiment.md). Evidence: [stats.json](stats.json).
The slice itself (`artifacts/data/ot3-slice-4k`, 359 MB) is regenerable with
`python -m mode_mlp.data --rows 4000 --shards 16`; it is kept for the GPU smoke and is
disposable once the training data decision produces a new slice.

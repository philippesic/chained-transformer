# Data pipeline

`python -m chained_transformer.data` turns a random sample of OpenThoughts3 into a labelled
training set, `stats.json` and `sample.jsonl`. Code: `chained_transformer/data.py` (sampling,
rendering, tokenizing, writing), `chained_transformer/mix.py` (row profiles, statistics, mix
policies), `chained_transformer/modes.py` (mode labels). Tests: `tests/test_data.py`,
`tests/test_mix.py`. [design.md](design.md#data-chained_transformerdatapy-chained_transformermixpy)
has the reasons; this page has the mechanics.

## Source

`open-thoughts/OpenThoughts3-1.2M`: 120 parquet shards with `conversations`, `domain` (code,
math, science) and `source` columns, among others. The rows are sorted by domain, so the first
rows of a few shards are not a sample; only row groups drawn from every shard are.

## Stages

1. **Sample.** Read the footer of all 120 shards and list every row group with its row count and
   the compressed size of the three columns used. Draw `--groups` distinct row groups (default
   400), weighted by rows, with `random.Random(--seed)`. Keep them in draw order until the next
   one would pass `--max-download-gb` (default 6), then sort by shard and group. Each chosen row
   group is fetched whole, so the printed `X GiB compressed` is the download.
2. **Read.** Take the first `--rows-per-group` rows (default 12) of each chosen row group.
3. **Render.** Keep single-turn rows (one user turn, then one assistant turn); others are
   counted as `skipped_not_single_turn` and dropped. Write the Qwen chat format by hand with no
   system prompt: `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n…<|im_end|>\n`, the
   assistant text verbatim, `<think>…</think>` included.
4. **Tokenize and label.** Tokenize with offsets. `loss_mask` is 1 from the first token of the
   assistant text to the end, `<|im_end|>\n` included. `mode_ids` holds each token's position
   mode from `ModeTracker`: prompt tokens are `think`; in the assistant turn the mode is `think`
   until `</think>`, then `answer`; a line that opens a code fence (three or more backticks)
   switches to `code` at the newline that ends it, and the matching closing fence line switches
   back to the mode outside the fence at its own newline.
5. **Mark completeness.** A row is `complete` when it has `</think>` and fewer than 16,000
   assistant tokens.
6. **Profile.** Every row becomes a `mix.RowProfile`: domain, total and assistant tokens,
   completeness, loss tokens per segment label, mode switches, code fences inside thinking and
   inside the answer, where the answer starts and where code blocks start (as fractions of the
   assistant turn).
7. **Select.** The policy picks the rows to keep (below). Only those are written.

A token's segment label is the position mode of the token before it: the mode the model is in
when it predicts that token. Training uses the same convention (`train.make_batch`), so loss by
label in `train.jsonl` and token shares in `stats.json` count the same thing.

## Mix policies

`mix.CANDIDATES`; `--policy` picks one, default `equal-domains-16k`.

| policy | rows eligible | domain weights |
| --- | --- | --- |
| `equal-domains-16k` | complete, at most 16,384 tokens | code 1, math 1, science 1 |
| `code-heavy-16k` | complete, at most 16,384 tokens | code 2, math 1, science 1 |
| `natural-16k` | complete, at most 16,384 tokens | none (the sample's own mix) |
| `equal-domains-8k` | complete, at most 8,192 tokens | code 1, math 1, science 1 |
| `natural-8k` | complete, at most 8,192 tokens | none |

Rows over the limit are left out, never cut. Selection shuffles the eligible rows with the
seed. With domain weights, it repeatedly takes the next row from the domain whose tokens so
far, divided by its weight, are lowest, and stops as soon as that domain has no rows left or
the token budget (`--token-budget`, default all eligible tokens) would be passed. So the
equal-domain policies stop at the scarcest domain's supply. Without weights, it takes
shuffled rows until the budget.

## Outputs

`--out` (default `artifacts/data/ot3-mix`) holds:

- The Arrow dataset (`datasets.load_from_disk`), one row per trace: `input_ids` (int32),
  `mode_ids` and `loss_mask` (int8), `domain`, `source`, `has_think_close`, `assistant_tokens`,
  `complete`.
- `stats.json`:
  - `sampling`: row groups read, `download_gib`, rows per group, seed and the list of
    `shard#group` read, enough to rebuild the same sample.
  - `sample_summary`: the whole sample before selection: rows and complete rows by domain,
    loss-token shares by label (all rows, complete rows, complete rows per domain), mode
    switches per complete row (mean and histogram), code blocks in thinking versus the answer
    and where they start, where the answer starts, length quantiles, and `length_caps` (what
    each length limit would keep).
  - `policies`: for every candidate policy, the rows, loss tokens and share per label it would
    select, and its rows per domain.
  - `chosen_policy` and `selected`: the policy used and what was written.
- `sample.jsonl`: the first 12 complete rows read (selected or not), each as runs of one
  segment label over the assistant turn, with a token count and the run's first 160
  characters, for reading the mode boundaries by eye.

## Using it

Train with `--seq-len` equal to the policy's limit (16,384 for the `-16k` policies), so no row
is cut; `train.py` still cuts longer rows and counts them as `truncated_rows`. Every row is
already complete, so `--complete-only` changes nothing. One row is one micro-batch.

The same `--seed`, `--groups`, `--rows-per-group` and `--max-download-gb` read the same row
groups and select the same rows. `--head` keeps the old layout (the first rows of evenly spaced
shards, no policy) for comparison with `ot3-slice-4k`.

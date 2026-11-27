# Eval suite notes

How the suite under `evals/` is built, what it was checked against, and what is still open.
Written during the overnight cloud run on 2026-11-26 (CPU-only container).

## Layout

- `types.py`: `Problem`, `Score`, `Loaded` and the `Benchmark` protocol every scorer implements
  (`load`, `score`, `reference_response`).
- `segments.py`: splits a response into think, code and answer text with the repo's
  `ModeTracker`, so scorers and leakage see the same segments the model was routed by.
- `math500.py`, `humanevalplus.py`, `livecodebench.py`: the three scorers.
- `sandbox.py`: subprocess runner for untrusted code (LiveCodeBench).
- `leakage.py`: thinking-leakage metrics.
- `generation.py`: loads a mode checkpoint, a plain checkpoint, a Hub name or three separate
  fine-tunes, generates through the mode hook (or the separate-models loop) and records tokens by
  segment label, switches, finish state, tokens/s and peak memory.
- `run.py`: the CLI. One `report.json` per output directory plus per-benchmark
  `responses.jsonl` and `scores.jsonl`.

`chained_transformer/evals.py` and its test `tests/test_evals.py` are the older stubs; nothing
else imports them. Delete both once this branch is merged.

## Running

On the 2080 Ti, in this project's pueue group `ct`:

```sh
pueue add -g ct -- uv run --locked --no-sync python -m evals.run --model <checkpoint or Hub name> \
  --benchmarks math500 humanevalplus livecodebench --out artifacts/evals/<name>
```

- Defaults: Qwen3's thinking-mode sampling (temperature 0.6, top-p 0.95, top-k 20), 32,768 new
  tokens, batch 1, bf16 on CUDA and fp32 on CPU. `--greedy` for deterministic runs; the separate
  arm only decodes greedily.
- `--separate THINK CODE ANSWER` runs the `separate` arm from three plain fine-tunes.
- GPU memory: Qwen3-0.6B's KV cache is 112 KiB per token in bf16, so 16 sequences of 8,192 new
  tokens need over 14 GiB, more than the 2080 Ti or the 5080 has left after the weights.
  On CUDA, `evals.run` checks the batch against free memory before generating (KV cache plus
  prefill attention, within 80% of what is free) and stops with the largest batch that fits.
  `--batch-size auto` picks that batch itself, up to 16; each benchmark's `perf.batch_size`
  records the batch used. It also sets the same allocator option as training
  (`expandable_segments:True`) unless one is set already.
- The `separate` arm re-prefills a model's missed tokens in chunks of 2,048 and keeps only the
  last position's logits, so a switch after thousands of thinking tokens does not materialise
  logits for every position.
- Batched decoding refills: when half the rows of a group have finished and others are
  waiting, the `generate` call stops and unfinished rows carry their tokens over and re-prefill
  (in chunks of 512 tokens). Rows run with others of the same length class, because a padded
  batch pays for its longest row on every row, in memory and in decode time. The budget is a
  token capacity, the batch that fits at full width times prompt plus `--max-new-tokens`: every
  call keeps rows times width inside it, so short rows run up to 16 at a time (`perf.refill_rows`)
  and long rows wait for a group of their own. Each call decodes into a cache allocated once at
  its final width. Greedy output is identical to decoding one problem at a time (tested).
  `--no-refill` keeps fixed batches; `perf.generate_calls` counts the calls. Unfinished rows
  waiting for a group are lost if the run is killed, like the rows of a running batch.
- `--stop-loops` (off by default) ends a response once its last 768 tokens repeat exactly with a
  period of at most 256, checked every 128 tokens, and records `stopped: "loop"`; it needs the
  refill decoder. It changes what a response is, so use one setting for every arm.
- `--sample N` or `--sample BENCH=N` evaluates a fixed subset: problems ordered by a salted hash
  of their ids, so a larger N keeps every problem of a smaller one. `--resume` keeps the complete
  rows of an earlier `responses.jsonl` and generates only the missing problems; it refuses when
  the earlier report used another model or other generation settings (batch size and refill
  aside), when the checkpoint's weights changed at the same path (a fingerprint of sampled
  bytes), or when responses exist with no report to check them against. The report is written
  before the first benchmark for that reason. Resumed benchmarks keep the earlier speed, batch
  size and memory figures and add the top-up's. Together they let a sample grow, or widen to the
  whole benchmark, without redoing work.
- `--attention auto` (the default) routes attention through the harness's fused path. On the
  2080 Ti (compute capability 7.5) PyTorch's flash and cuDNN kernels need 8.0 or later and its
  memory-efficient kernel takes only fp16 and fp32 there, so plain bf16 attention would fall
  back to the math kernel; the fused path upcasts to fp32 instead. `--attention sdpa` or `eager`
  pick Hugging Face's own.
- `--attention lean` is the fused path, except that one-token decode steps are computed over the
  grouped cache in fp32 without first repeating it to 16 heads. Same arithmetic up to summation
  order; in bf16 a long greedy response can still part from the `auto` one, so every arm of a
  comparison uses one choice (the resume check enforces it per directory). Not yet run on a GPU:
  `python -m tools.decode_bench` times a decode step for each attention and cache choice and
  reports how often they pick the same token. See
  [the decode-cost result](../results/eval-decode-cost/result.md).
- `--limit N` marks the report as a smoke run. `--score-only` rescores saved responses and keeps
  the earlier report's sample, speed figures and thinking setting.
- With `--no-thinking` the chat template puts an empty think block in the prompt, so responses
  hold no think tags; scoring, leakage and `think_closed` read each response as that full turn.
  `--oracle` scores each benchmark's reference responses with no model, to check a scorer.
- `--source math500=hf|github|<jsonl>`, `--source livecodebench=<dir of test*.jsonl>`; without a
  source, MATH500 and LiveCodeBench download from the Hugging Face Hub.
- `tools/real_model_smoke.py` runs the morning checks on a machine with Hub access.

## Scorers

- MATH500: Qwen's instruction ("Please reason step by step, and put your final answer within
  \boxed{}."). The primary score reads only the answer segment: the last `\boxed{}` there,
  checked with math-verify against `$gold$`, with a fallback to the whole answer segment when
  there is no box. Unclosed thinking scores 0 on the primary metric; `accuracy_lenient` also
  extracts from the full text when thinking never closed. math-verify runs in killable worker
  processes. Sources: `HuggingFaceH4/MATH-500`, or OpenAI's original
  `prm800k/math_splits/test.jsonl` at commit `7ecc794`, which is a Git LFS object (content
  sha256 `35dc4108…`).
- HumanEval+: evalplus 0.3.1, dataset `v0.1.10`, evalplus's own chat instruction, last fenced
  Python block after thinking, evalplus's sanitizer, then `python -m evalplus.evaluate` in a
  subprocess with a 1 h cap. `accuracy` is base and plus passing; `pass_at_1_base` is HumanEval.
- LiveCodeBench: mirrors `lcb_runner` at commit `28fef95` (2025-07-15): the `CodeQwenInstruct`
  prompt (the style lcb_runner assigns to Qwen3), last fenced block after thinking, the official
  stdin and functional grading with a 6 s per-test alarm, first-failure stop, and the official
  overall budget. Default problems: `release_v6`, contest dates 2024-10-01 to 2025-05-01, 341
  problems. The 166 before 2025-02-01 are the window of the Qwen3 technical report (`release_v5`),
  too few on their own for the kill rule's sample. Private tests are decoded with a restricted
  unpickler that refuses every global.
- Leakage: the existing think-word list ("wait", "hmm", "let me", "alternatively", "actually",
  "double-check", "I think"), per 1,000 answer words, over responses whose thinking closed; plus
  the share of answers with any think word, the same rate inside thinking for contrast, and code
  characters on each side of `</think>`.

## Checks run here

- Unit tests for every scorer, the sandbox, segments, leakage and the CLI (CPU, no network).
- HumanEval+ oracle: 163/164 canonical solutions pass base and plus; bodies replaced by
  `return None` pass 0/164. The one failure, `HumanEval/32`, is an evalplus 0.3.1 bug (its
  `find_zero` oracle never records a pass), so 163/164 is the ceiling for every arm. Details in
  `results/eval-scorer-checks/`.
- CLI plumbing: a tiny random mode checkpoint with a toy tokenizer (`tools/tiny_checkpoint.py`)
  through generation, mode tracking, scoring, `--score-only` and the separate arm. The toy
  tokenizer keeps `<think>` and `</think>` when decoding, as Qwen3's is believed to (not checked
  here), so CPU runs score responses whose thinking closed.

## Blocked here, and the defaults taken

- Hugging Face is blocked by this container's network policy (huggingface.co and its CDN).
  So no Qwen3-0.6B weights or tokenizer, no `HuggingFaceH4/MATH-500`, no LiveCodeBench data.
  The two tokenizer tests in `tests/test_data.py` fail here for the same reason.
- No base-model smoke numbers yet. `tools/real_model_smoke.py` produces them on a machine with
  Hub access.
- MATH500 oracle over all 500 problems not run: the GitHub copy is an LFS object served from
  media.githubusercontent.com, which is also blocked. A six-problem local file scored 6/6.
- LiveCodeBench: the `release_vN` to file mapping (vN = `test.jsonl` … `testN.jsonl`) comes from
  the dataset's loader script on the Hub, which could not be read. The report records the
  README's count for the release next to the count loaded; v6 gives 1,055 (checked 2026-11-27).

- The pilot evaluates each arm as soon as its training runs are done (`commands.sh` and `run`
  use that order), so an eval problem shows up after the first arm, and uses `--batch-size auto`
  by default. The `separate` arm goes last, after both seeds of the other arms: it is not in the
  kill rule and decodes at batch 1, so its evals are the slowest step of the pilot.
- The pilot (`tools/pilot.py`) evaluates every arm greedily, because the `separate` arm's
  re-prefill loop only decodes greedily; Qwen3's recommended sampling (0.6/0.95/20) is for later
  runs. Its defaults (1,000 steps, 4,096 tokens, 8 rows per step, learning rate 1e-5 and 1e-4
  for `lora`, complete rows only) are placeholders until decisions 1 and 2 in the task are
  settled; a plan's config is fixed once written.
- The pilot trains through `train.main`'s `on_model` hook, which exists only on `cloud/harness`.
  On a tree with both branches merged, a 7-run tiny pilot on CPU trained, saved, evaluated and
  compared end to end.
- `tools/pilot_report.py` turns a finished pilot into the paper's Tables 1, 2, 4 and 5 and the
  kill-rule verdict (from `pilot.verdict`), tested on fake reports and run on the tiny CPU pilot.
  Its notes flag missing reports, smoke limits, CPU timings, and a `separate` arm decoded at
  batch 1 next to batched mode arms, which makes Table 5's tokens/s compare batch sizes.
- The verdict and Table 1's mean average the planned benchmarks; an arm missing one is
  incomplete. An eval counts only when it finished under the current plan and after its
  checkpoint's last training; others are listed as not counted (failed, an earlier sample, or
  retrained since), and a retrained arm's old eval directory is moved aside before it runs again.
- `tools/eval_power.py --odds` gives the chance of each verdict by true gap. Because every seed
  must clear 2 points, a true 2-point gap continues about 11% of the time at any sample size
  (each seed is a coin flip on each margin), a 4-point gap about 86% at the planned standard
  error, and a third seed makes the rule stricter rather than more sensitive. Its last columns
  give the rule the sample was sized for, both seed-mean margins clearing a one-sided 5% test:
  about 69% at a true 2-point gap with two seeds and 86% with three. Training noise between
  seeds raises that rule's false continues, since its line counts only sampling error.
  `pilot plan --kill-rule seed-mean` applies it; `every-seed`, the rule as written, is the default.
- The pilot defaults to the verdict eval set: `sft`, `full`, `lora` and `moe` on a per-benchmark
  sample sized by `tools/eval_power.py` for a 2-point gap at two seeds, assuming the arms
  disagree on 10% of problems (383 MATH500, all 164 HumanEval+, all 341 LiveCodeBench). After each
  eval, `pilot run` measures how often `full` and `sft`/`moe` disagree on shared problems (once
  100 are shared) and grows the sample if needed, never shrinking it; earlier evals resume to
  match. HumanEval+ has only 164 problems, so under the macro mean of three benchmarks the
  sample reaches the whole suite once the arms disagree on 15% or more. `--eval-set full` plans
  everything at once; `pilot widen` switches a verdict plan to it later. New tokens default to
  `--seq-len`, the longest row the arms trained on, for every arm alike.
- `tools/pilot_budget.py estimate` puts the verdict at about 4 to 7 days on the 2080 Ti (was 7 to
  15) and 1 to 2 days on the 5080, now mostly training on the 2080 Ti. These are estimates from
  stated assumptions until `measure` runs on each GPU.
- `evals/switch_stats.py` reads `mode_runs`, which responses record from this branch on; older
  `responses.jsonl` files count as responses without a trace.

## Deliberate differences from the official harnesses

- LiveCodeBench: `for line in sys.stdin` works. The official mock stdin (since 2025-05-23) has no
  `__iter__`, so that idiom raises TypeError there and fails correct programs. Other official
  quirks are kept: no float tolerance, `open(0)` reads empty input, `sys.stdin.buffer.readline()`
  always returns the first line.
- LiveCodeBench: code inside thinking is never graded; the window end date is exclusive;
  problems are sorted by date then id; the sandbox adds a 4 GB address-space limit, CPU and file
  size limits and kills the process group.
- The sandbox runs Python with `-I`, which ignores `PYTHONHASHSEED`, so set iteration order can
  vary between runs, as in the official harness.
- Sandbox limits: no network or filesystem isolation beyond a fresh working directory, no
  process-count limit, and a child that calls `setsid` escapes the group kill. Fine for scoring
  our own models' output on a research box; not a security boundary.

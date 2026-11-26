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

```sh
uv run --locked --no-sync python -m evals.run --model <checkpoint or Hub name> \
  --benchmarks math500 humanevalplus livecodebench --out artifacts/evals/<name>
```

- Defaults: Qwen3's thinking-mode sampling (temperature 0.6, top-p 0.95, top-k 20), 32,768 new
  tokens, batch 1, bf16 on CUDA and fp32 on CPU. `--greedy` for deterministic runs; the separate
  arm only decodes greedily.
- `--separate THINK CODE ANSWER` runs the `separate` arm from three plain fine-tunes.
- `--limit N` marks the report as a smoke run. `--score-only` rescores saved responses.
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
  overall budget. Default problems: `release_v5`, contest dates 2024-10-01 to 2025-02-01, the
  window of the Qwen3 technical report. Private tests are decoded with a restricted unpickler
  that refuses every global.
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
  through generation, mode tracking, scoring, `--score-only` and the separate arm.

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
  README's count for the release next to the count loaded; v5 should give 880.

- The pilot (`tools/pilot.py`) evaluates every arm greedily, because the `separate` arm's
  re-prefill loop only decodes greedily; Qwen3's recommended sampling (0.6/0.95/20) is for later
  runs. Its defaults (1,000 steps, 4,096 tokens, 8 rows per step, learning rate 1e-5 and 1e-4
  for `lora`, complete rows only) are placeholders until decisions 1 and 2 in the task are
  settled; a plan's config is fixed once written.
- The pilot trains through `train.main`'s `on_model` hook, which exists only on `cloud/harness`.
  On a tree with both branches merged, a 7-run tiny pilot on CPU trained, saved, evaluated and
  compared end to end.
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

# Eval decoding copies the whole cache every step; a leaner path gives the same answers

**Conclusion.** The eval's decode step costs time in proportion to the cached tokens in the
batch (rows times padded width), because every step copies the key/value cache several times.
On CPU with the real Qwen3-0.6B, removing two of those copies makes a step 3.9 to 6.0 times
faster and picks the same token at every step, and the leaner path reproduces the current
path's answers exactly on every case tried. The size of the gain on the RTX 2080 Ti is
**unresolved**: nothing here ran on a GPU. `tools.decode_bench` measures it in about four
minutes with the card free.

Where the copies come from, per decode step and per layer (read from the pinned sources):

- `DynamicCache` concatenates the layer's keys and values with the new token, rewriting both.
- `fused_sdpa` repeats keys and values from 8 to 16 heads, and below compute capability 8.0
  converts both to fp32, before the attention kernel reads them once more.

That is about 60 KB of memory traffic per cached token per layer against 4 KB of cache in
bf16. The branch changes three things:

- `--attention lean` computes one-token decode steps over the grouped cache in fp32: no repeat,
  and the fp32 conversion is of 8 heads, not 16. Prefill and training keep the fused path.
- The refill decodes into a cache allocated once per call at its final width, appending in place.
- The refill groups rows by length class inside the same token capacity (rows that fit at full
  width, times prompt plus max-new-tokens), so short rows run up to 16 at a time and long rows
  wait for a group of their own instead of padding every row to their width.

Decode step time, CPU, fp32, 6 rows, 6 threads (`decode-bench-cpu.json`):

| Attention, cache | 1,024 tokens | 3,072 tokens | Per 1,000 cached tokens | Same argmax as the first row |
| --- | ---: | ---: | ---: | ---: |
| `auto`, `DynamicCache` (current) | 977 ms | 2,771 ms | 146 ms | reference |
| `auto`, preallocated | 843 ms | 2,047 ms | 98 ms | 100% |
| `lean`, `DynamicCache` | 400 ms | 1,148 ms | 61 ms | 100% |
| `lean`, preallocated | 252 ms | 459 ms | 17 ms | 100% |

Largest logit difference from the current path: 4e-5.

Same answers (`identity-checks.json`):

| Case | Answers | Identical |
| --- | ---: | ---: |
| Base model, MATH500, to completion (657, 1,050 and 1,057 tokens): current path against branch | 3 | 3 |
| Rehearsal `full` checkpoint, 6 problems per benchmark, 768 tokens: current path against branch, text and mode runs | 18 | 18 |
| Base model, 7 MATH500 and 5 LiveCodeBench, 300 tokens: grouped refill with 6 rows in a capacity sized for 2, against one problem at a time | 12 | 12 |

**What it means for the pilot (forecast, not measured).** Two timings from earlier GPU logs fit
the same form: 36 to 43 ms per step at batch 1 with short contexts (`gpu_equivalence`) and 43
to 58 ms at batch 3 with 3,000 to 6,000 cached tokens (the real-model smoke). A line through
them is about 38 ms plus 2.7 ms per 1,000 cached tokens. `tools.eval_sim` replays both
schedulers over a set of response lengths under such a model, for 830 problems at a cap of
8,192 new tokens:

| Share of answers that hit the cap | Current path | Grouped refill only | Branch (slope 0.9 ms assumed) |
| ---: | ---: | ---: | ---: |
| 5% | 17 h | 9 h | 5 h |
| 15% | 23 h | 13 h | 7 h |
| 30% | 27 h | 16 h | 8 h |
| 45% | 31 h | 20 h | 10 h |

Hours per arm; the pilot has eight evals. The 0.9 ms slope assumes the GPU saves the same
share of traffic as the byte count suggests (60 KB to 20 KB); the CPU saved more.

**Base-model answer lengths.** Untrained Qwen3-0.6B, greedy, thinking on, cap 8,192 new
tokens, on CPU with the branch's path (`base-answer-lengths.jsonl`). The samples are the first
problems of the pilot's hash-ranked sample that had finished when the runs were read; answers
still being generated are left out, and those are long ones, so cap shares are if anything
low.

| Benchmark | Answers | Finished by 2,048 | by 4,096 | by 8,192 | Hit the cap | Tokens spent in capped answers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MATH500 | 23 | 39% | 61% | 70% | 7 (30%) | 63% |
| HumanEval+ | 9 | 56% | 78% | 100% | 0 | 0% |
| LiveCodeBench | 6 | 0% | 33% | 33% | 4 (67%) | 83% |

Every capped answer was still inside its thinking, and none of the 7 capped MATH500 answers
scored. MATH500 accuracy on the 23 is 14 (61%): 9 of 9 that finished within 2,048 tokens, 3 of
5 between 2,049 and 4,096, and 2 of 2 after 4,096. The loop detector behind `--stop-loops`
fires on 3 of the 11 capped answers and on none that finished; it would save 2% of MATH500
tokens and 11% of LiveCodeBench tokens. Long answers here are open-ended thinking, not exact
repeats, so `--stop-loops` buys little.

Replaying the pilot's eval mix with these lengths (500 MATH500, 164 HumanEval+ and 166
LiveCodeBench drawn with replacement, 33% at the cap, mean 4,244 tokens) under the forecast
step-time model:

| Cap | Current path | Branch (slope 0.9 ms assumed) |
| ---: | ---: | ---: |
| 8,192 | 28 h per arm | 9 h per arm |
| 4,096 | 10 h per arm | 4 h per arm |

An answer's cost grows with the square of its length, which is why capped answers dominate.
Fine-tuned arms were trained on traces averaging about 5,000 tokens, so they are likely to
run longer than the base.

**Conditions.** Branch `overnight/eval-speed` (lean attention 3de8321 and 47a4311, grouped
refill 27a275e, preallocated cache cac168a, simulator 17fc7ca), CPU only on the 2080 Ti host
while the pilot trained. The synthetic lengths behind the first forecast table are log-normal with a
median of 1,600 to 2,000 tokens plus the stated share at the cap.

**Limitations.** No GPU measurement of step time, memory or token agreement. In bf16 on the GPU
the current path and `lean` both compute attention in fp32 but round differently in places, so
long greedy answers can diverge between them, as they already can between batch sizes; all
arms must therefore use one attention choice. The forecast's slope comes from two coarse GPU
timings. The grouped refill parks unfinished rows while other groups run, so a crash loses up
to a few dozen partly generated answers, not the batch's seven.

**Disposition.** Unresolved until `python -m tools.decode_bench --out artifacts/bench/2080.json`
runs on the 2080 Ti. Adopt `--attention lean` for the pilot's evals only if `lean` with the
preallocated cache is at least 1.5 times faster per step than the current path at 7x4096 and
7x8192 and `same_argmax_rate` is at least 0.99; whether to pause the pilot for that check is
Philip's decision. `--stop-loops` stays off: the base model's long answers are not
repeats. Whether to lower the cap is better judged from the first eval's own answers than
from the base model. The length runs' outputs (`artifacts/lengths` in the overnight worktree)
are disposable.

# Eval scorer oracle checks (2026-11-26, CPU cloud container, 4 cores)

Oracle runs feed each scorer responses built from the gold answer or canonical solution, plus
negative controls, to check extraction and scoring end to end without a model.

## HumanEval+ (`humanevalplus.json`)

Command: `python -m evals.humanevalplus --workers 4 --out results/eval-scorer-checks/humanevalplus.json`

Data: evalplus 0.3.1, HumanEvalPlus `v0.1.10` from
`github.com/evalplus/humanevalplus_release` (164 tasks, file sha256
`42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`, evalplus md5
`fe585eb4df8c88d844eeb463ea4d0302`).

| control | passed (base and plus) | base only | plus only | fenced block found | `score` wall time |
| --- | --- | --- | --- | --- | --- |
| canonical solutions (prompt + canonical body after an empty think) | 163/164 | 163/164 | 163/164 | 164/164 | 22.0 s |
| bodies replaced by `return None` | 0/164 | 0/164 | 0/164 | 164/164 | 22.0 s |

The whole command took 45 s with evalplus's ground-truth cache already built (building it the
first time added about 10 s). About 20 s of each `score` call is evalplus waiting for its own
"stuck checker" thread, which only exits after a 20 s sleep, so `score` never returns in under
about 20 s.

The one canonical failure, `HumanEval/32` (`find_zero`), is an evalplus 0.3.1 evaluator bug, not
an extraction or sanitizer problem: its special oracle for `find_zero` runs
`assert abs(_poly(*inp, out)) <= atol` and then `continue`s before recording the test as passed,
so `len(details) != len(inputs)` and every solution is marked `fail`. Running the unsanitized
`prompt + canonical_solution` through `python -m evalplus.evaluate` directly gives the same
`fail`/`fail` with no failing test listed. With evalplus 0.3.1 the ceiling is therefore 163/164;
fixing it needs a newer evalplus or a special case for that task.

The negative control shows the evaluator rejects a function with the right signature but no
body for every task, so the 163 passes are not an artefact of extraction or sanitizing.

## MATH-500: not run here

The `github` source could not be fetched in this container. OpenAI's
`prm800k/math_splits/test.jsonl` at commit `7ecc794703b2877f63226f2477a49b34f9b25163` is stored
with Git LFS: raw.githubusercontent.com serves only the LFS pointer (sha256
`bf4bf2b73edea199aed01c5b3bdc90404d7f2419daf6931fc9a79622fcbf0cf2`, verified here), which names
the content as sha256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`,
446564 bytes. The content itself is served from media.githubusercontent.com, which this
environment's proxy refuses (HTTP 403), and huggingface.co is blocked by policy, so the 500
problems were not available. `evals/math500.py` pins both hashes and downloads the pointer and
then the content, checking both.

To run the three controls (reference answers, every problem given the next problem's answer, and
the box only inside unclosed thinking) where that host is reachable:

`python -m evals.math500 --source github --workers 4 --out results/eval-scorer-checks/math500.json`

What was checked instead: unit tests (`tests/test_math500.py`) on hand-written problems cover
nested braces, several boxes, boxes inside thinking being ignored, unclosed thinking (primary
wrong, lenient right), math_verify equivalences (fractions vs decimals, `\dfrac`, tuples with
`\left(`, `\text{(C)}` vs `C`, surds, `x = 5` vs `5`, percent and dollar signs, intervals,
complex numbers) and non-equivalences, plus the oracle command on a six-problem local jsonl
(reference 6/6, shifted answers 1/6 where two problems share the answer `10`, box only in
unclosed thinking 0/6 primary and 6/6 lenient).

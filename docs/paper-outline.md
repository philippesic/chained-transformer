# Paper outline (draft, 2026-11-26)

Nothing below has run on real weights yet. Numbers from other papers are as recorded in the
README's prior-art section and the project's prior-art notes (checked 2026-11-26); recheck each
against the paper before citing.

## Claim

One Qwen3 decoder keeps attention, embeddings, norms and the LM head shared, and holds three
MLP sets per layer: `think`, `code` and `answer`. The model enters a mode through tokens it writes
itself (`</think>`, opening and closing code fences) and every mode reads and writes one KV cache.

1. **Quality.** At equal training tokens, hard-routed full modes beat a plain fine-tune (`sft`)
   and a same-size learned per-token MoE (`moe`) by at least 2 points on the mean of MATH500,
   HumanEval+ and LiveCodeBench, on both of two seeds. If not, the idea is dropped (the kill rule
   fixed before any run).
2. **Upgrades.** One mode can be retrained alone with everything else frozen. Positions before
   the first switch into that mode are bit-identical by construction; the claim to test is that
   the other modes' benchmark scores and per-label loss hold after it.
3. **Cost.** A switch costs nothing beyond loading another MLP set: no re-prefill, one prefix
   cache, the same active parameters per token as the base model.

What would sink each claim: (1) `full` within 2 points of `sft` or `moe`, or `lora` tying `full`
(then the result is "LoRA modes", a weaker claim); (2) `answer` and `think` scores dropping after
a code upgrade by more than continued `sft` training drops them; (3) mode switches misfiring
(fences written in prose), which shows up in the switch statistics.

## Related work

| Work | Shares with us | Difference |
| --- | --- | --- |
| Path-Lock Expert (2604.27201), Metis-HOME (2510.20519) | think/no-think MLP sets, shared attention | two sets, locked for a whole sequence; ours switch mid-answer, three sets |
| Staged Decoding (2602.24210) | swaps a LoRA at `</think>` inside one response | LoRA, reported MathIF drop 45 to 28 |
| MixReasoning (2510.06052), WeAct (2610.26847) | switches mid-response, per-mode training | per-mode caches with re-prefill (MixReasoning) or KV rebuilt at each switch (WeAct); our `separate` arm is this baseline |
| aLoRA (2504.12397) | adapter switched mid-sequence, reads the base model's KV | LoRA over a frozen base; ours trains every mode for the shared cache |
| PA-MoE (2602.17038) | LoRA experts chosen per agent step | step-level choice among LoRAs |
| DEMix (2108.05036), BTX (2403.07816), MiCRo | domain or skill MLPs with shared attention | routing by domain or learned per token; ours by self-emitted segment tokens |
| FlexOlmo, BAR (2604.18473) | upgrading one expert alone | BAR: code expert +16.5 on code with others unchanged, but needs most experts active |
| S-LoRA, Punica | serving many adapters over one base | adapters, batched kernels; ours swaps whole MLP sets at switch tokens |
| SAEM (2608.21614), Stepped MoE (2610.07348), Task-MoE (2110.03742) | expert residency at stage boundaries | serving-side; we need it for parking idle modes |

Evidence against, which the paper has to answer head-on:

- BTX: learned per-token routing beat hard domain routing, 47.9 vs 43.4. Our `moe` arm repeats
  this comparison at small scale with segment routing instead of domain routing.
- MiCRo: an FFN-only variant with shared attention did not specialise; only full-block clones
  did.
- Thinking Leakage (2610.14270): the think/no-think signal sits in a few attention heads and a
  small residual subspace, not in the FFNs. If so, MLP modes may not separate thinking from
  answering; the leakage metric tests this directly.
- LoRA Without Regret (Thinking Machines, 2025): LoRA on MLPs roughly matches full fine-tuning at
  SFT scale, which predicts `lora` ties `full`.
- DroidSpeak: naively reusing a KV cache across fine-tunes loses over half the accuracy, which
  is why modes must be trained together on one cache rather than spliced from separate
  fine-tunes (our `separate` arm re-prefills instead).

## Experiments

1. **Main comparison** (`tools/pilot.py`). Qwen3-0.6B, five arms (`sft`, `full`, `lora`, `moe`,
   `separate`), seeds 0 and 1, the same rows in the same order, equal steps and sequence length.
   Data: complete OpenThoughts3 rows under the chosen mix policy. Evals: MATH500 (answer segment,
   math-verify), HumanEval+ (evalplus 0.3.1), LiveCodeBench `release_v5` with contest dates
   2024-10-01 to 2025-02-01, greedy for every arm because `separate` decodes greedily. The pilot's
   verdict runs on a sample sized for a 2-point gap (`tools/eval_power.py`); paper numbers come
   from `pilot widen`, which completes the full suite and adds `separate`.
2. **Single-mode upgrade** (`chained_transformer.upgrade`). Retrain `code` on a code-heavy slice
   with the trunk and other modes frozen; report benchmark deltas for all three, per-label KL from
   the parent and drift before and after the first code position. Run it with loss on code
   tokens and with loss on whole responses: in the CPU toy
   ([results/upgrade-demo](../results/upgrade-demo/result.md)) a code-only loss left the frozen
   `answer` mode answering for the old code. Baselines: continued `sft` on the same rows (all
   parameters, and MLPs only), and upgrading one `moe` expert.
3. **Leakage** (`evals/leakage.py`). Think-words per 1k answer tokens and unclosed-think rate by
   arm; this is where Thinking Leakage's result would show.
4. **Switch statistics** (`evals/switch_stats.py`). Switches per response, transitions, run
   lengths, where each mode first appears, and pass rate by switch count; training data against
   each arm's generations.
5. **Serving cost.** Tokens/s, peak memory and re-prefilled tokens for modes against `separate`
   from the eval reports; the vLLM path from `cloud/serving` later.
6. **Ablations.** `lora` rank; `moe` top-1 (same active compute as `full`) against top-2; prompt
   tokens in `think` against `answer`; Qwen3-1.7B (configs in `configs/qwen3-1.7b/`).

## Expected tables and figures

- **Table 1, main results.** Rows: the five arms. Columns: MATH500, HumanEval+ (base and plus),
  LiveCodeBench, mean, answer leakage per 1k tokens, tokens/s, peak memory, total and active
  parameters. Mean over two seeds with both values shown.
- **Table 2, kill rule.** Per seed: `full - sft`, `full - moe`, `lora - full` in points
  (`python -m tools.pilot compare` writes these).
- **Table 3, upgrade.** Before and after a code upgrade: each benchmark, per-label KL from the
  parent, drift before the first code position (zero) and after it; rows for the mode upgrade and
  each baseline.
- **Table 4, switching.** Data against each arm: switches per response, transition shares, code
  and answer run lengths, share of responses that reach each mode.
- **Table 5, serving.** `separate` against modes: re-prefilled tokens per response, tokens/s.
- **Figure 1:** the architecture with one cache and per-mode MLPs. **Figure 2:** a real response
  coloured by mode. **Figure 3:** pass rate against number of switches, per arm.

`python -m tools.pilot_report --root artifacts/pilot/NAME` prints Tables 1, 2, 4 and 5, the pass
rate by switches for Figure 3 and the kill-rule verdict, and writes `results.md` and `results.json`.

## Evaluation section, paragraph by paragraph

1. **Benchmarks and scoring.** MATH500 scores only the answer after `</think>` with math-verify
   (the lenient score over the whole response is a leakage diagnostic, not a result);
   HumanEval+ with evalplus 0.3.1, plus as the primary and base alongside; LiveCodeBench
   `release_v5`, contest dates 2024-10-01 to 2025-02-01 (880 problems), stdin and functional tests
   in a sandbox. The primary metric is mean accuracy over the three, in points. HumanEval+ grades
   code inside unfinished thinking and the other two do not, so give each arm's unclosed-think rate
   beside its scores.
2. **Decoding.** Greedy for every arm, the same chat template with thinking on, at most the
   training sequence length in new tokens. Batched and one-at-a-time greedy outputs match on CPU
   (tests); bf16 on the GPU can differ by batch, so say which batch each arm ran at.
3. **Verdict sample, then the full suite.** The kill rule was judged on a fixed hash-ranked
   sample sized for a per-seed standard error of 1.14 points at an assumed discordance of 0.10
   (361 / 164 / 361 problems), grown when the first arms disagreed more. The tables use the full
   suite after `pilot widen`. Report both, and that the verdict came before widening.
4. **The rule and its odds.** State the rule exactly as fixed, then its operating
   characteristics from `python -m tools.eval_power --odds`: requiring both margins to clear
   2 points on every seed, with a per-seed SE of 1.14 and the two margins correlated at about 0.5,
   a true 2-point gap gives continue 11%, mixed 44%, kill 44%; a 3-point gap continues 49%, a
   4-point gap 86%. More problems do not change the 2-point case, and a third seed makes the rule
   stricter. A reader will compute this, so the paper says it first.
5. **Uncertainty.** Each margin gets a paired standard error from per-problem pass or fail,
   pooled over benchmarks (Table 2). Two seeds give no seed-level interval, so both seeds are
   shown everywhere.
6. **Fairness.** Same rows in the same order, equal steps and tokens per arm; total and active
   parameters for each arm; `moe` at top-2 and top-1; `separate` re-prefills at each switch.
7. **Hardware.** Which GPU, bf16 with fp32 attention on cards older than sm80, and that speed
   compares only at equal batch size.

## Results section, paragraph by paragraph

Every number in the text comes from `results.json` (`tools/pilot_report.py`) and appears in a
table.

1. **Verdict.** The first sentence is continue, mixed or kill, with `full - sft` and
   `full - moe` per seed and their standard errors (Table 2).
   - Continue: claim 1 holds at this scale; go on to the upgrade (paragraph 6).
   - Mixed: give both seeds and say the rule does not decide. What a mixed result means was not
     fixed before the runs; decide it before the pilot (a third seed under a pooled rule, or
     treat it as kill).
   - Kill: say so plainly. The paper becomes a negative result: which arm matched `full`, by how
     much, and whether that held on both seeds.
2. **LoRA.** `lora - full` per seed; within 1 point on every seed means the result is LoRA modes.
3. **Where the gain is.** Table 1 by benchmark: math against code, answer leakage per arm, and
   whether the gain survives in the unclosed-think rate.
4. **Switching.** Table 4 and Figure 3: do the arms switch like the data, pass rate by number of
   switches, and fences opened in prose.
5. **Serving.** Table 5: re-prefilled tokens and tokens/s, modes against `separate`, at matched
   batch size.
6. **Upgrade.** Table 3, only after a continue: the retrained mode's gain and the other modes'
   change against continued `sft`.
7. **Sample against full suite.** If the full-suite margins would have given a different verdict
   than the sample did, say so and give both.

## Threats to validity

- Equal tokens is not equal parameters: `full` holds 1.89 times the base parameters with the
  same active compute; `moe` top-2 doubles active MLP compute. Report both counts and run top-1.
- `code` and `answer` get about 1% and 4% of loss tokens in the first slice, so those modes see
  little data; the mix policy and per-label losses must be reported.
- Self-emitted routing can misfire. Count fences in think text and switches that never close.
- Small scale (0.6B, two seeds) and greedy decoding; say so and add sampled runs if the result
  holds.
- The kill rule needs every seed to clear the line, so at a true gap of exactly 2 points it
  continues about one time in nine; report its odds (`eval_power --odds`) next to the verdict.
- The upgrade's zero drift before the first switch is structural (causal attention), not a
  finding; the result is the drift after it.

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
   2024-10-01 to 2025-02-01, greedy for every arm because `separate` decodes greedily.
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

## Threats to validity

- Equal tokens is not equal parameters: `full` holds 1.89 times the base parameters with the
  same active compute; `moe` top-2 doubles active MLP compute. Report both counts and run top-1.
- `code` and `answer` get about 1% and 4% of loss tokens in the first slice, so those modes see
  little data; the mix policy and per-label losses must be reported.
- Self-emitted routing can misfire. Count fences in think text and switches that never close.
- Small scale (0.6B, two seeds) and greedy decoding; say so and add sampled runs if the result
  holds.
- The upgrade's zero drift before the first switch is structural (causal attention), not a
  finding; the result is the drift after it.

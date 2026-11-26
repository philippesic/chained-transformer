# mode-routed-mlp

A decoder-only transformer whose attention, embeddings, norms and LM head are shared, while
every layer holds several MLP sets called **modes**: `think`, `code` and `answer`. The model
switches mode by itself inside one answer, at boundaries marked by tokens it writes: `</think>`
ends thinking, and opening or closing a code fence enters or leaves `code`. All modes read and
write **one shared KV cache**, so a switch never re-encodes the context. Every mode starts as a
copy of one base model's MLPs.

Goals:

1. Beat a plain fine-tune and a same-size per-token MoE at equal training tokens.
2. Show one mode can be retrained alone, with the shared trunk frozen, without hurting the others.
3. Later: a vLLM or llama.cpp path that swaps mode MLPs at switch tokens, keeps one prefix cache
   and can park idle modes in CPU RAM.

Status: setup. The model surgery, mode tracking, data slice, training loop and eval stubs exist
and pass CPU tests; GPU qualification is pending. See [the task](tasks/first-experiment.md).

## Prior art (checked 2026-11-26)

Already done:

- Two think/no-think MLP sets with shared attention, locked for a whole sequence: Path-Lock
  Expert (arXiv 2604.27201), Metis-HOME (2510.20519).
- LoRA swaps in the middle of an answer: Staged Decoding (2602.24210), MixReasoning
  (2510.06052), WeAct (2610.26847; plan/search/write LoRAs that rebuild the KV cache at each
  switch).
- Domain FFNs with shared attention: DEMix (2108.05036). Single-expert upgrades: FlexOlmo,
  BAR (2604.18473).
- Expert residency or offloading at stage boundaries: SAEM (2608.21614), Stepped MoE
  (2610.07348), Task-MoE (2110.03742).

Evidence against the idea:

- Hard domain routing lost to learned per-token routing in BTX (47.9 vs 43.4; 2403.07816).
- MoE experts specialise by syntax, not topic (2604.09780).
- The think/no-think signal sits mostly in attention heads (2610.14270).
- LoRA on MLPs roughly matches full fine-tuning at SFT scale (Thinking Machines, "LoRA Without
  Regret").
- Naively reusing a KV cache across fine-tunes loses over 50% (DroidSpeak).

Still open: three or more full-MLP modes that switch mid-answer on one shared cache trained
for it; single-mode upgrades in that setup; mode routing combined with looping.

## First experiment

Base model `Qwen/Qwen3-0.6B`: 596M parameters, 44% of them in MLPs. Three full modes total
1.12B parameters; three rank-64 LoRA modes total 662M. Training data:
`open-thoughts/OpenThoughts3-1.2M`, with every token labelled think, code or answer. Five arms train on the same tokens:

| Arm | What changes | Routing |
| --- | --- | --- |
| `sft` | Plain fine-tune of the base | none |
| `full` | Three full MLP copies per layer | hard, by segment label |
| `lora` | Shared base MLP plus one LoRA per mode | hard, by segment label |
| `moe` | Three MLP copies as experts with a learned top-k router (BTX-style) | learned, per token |
| `separate` | One fine-tune per mode, switched mid-answer with re-prefill | by segment, separate caches |

Metrics: MATH500, HumanEval+, a LiveCodeBench subset, think-word leakage into answers, peak
memory and tokens/s.

Decision rules: drop the idea if `full` does not beat both `sft` and `moe` by about 2 points
across two seeds. If `lora` ties `full`, continue with LoRA modes.

## Setup and navigation

```sh
uv sync --locked --extra eval
git config core.hooksPath .githooks
uv run --locked --no-sync python -m tools.check
```

- [Design](docs/design.md): how modes, routing, data, training and evals work.
- [Development](docs/development.md): environment, checks and GPU qualification.
- [Task](tasks/first-experiment.md): current state, open decisions and next steps.
- [FINDINGS](FINDINGS.md): index of results.

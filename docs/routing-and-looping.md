# Mode routing plus looping (design note)

Status: proposal written 2026-11-26 in the cloud harness thread. Nothing here is built or
measured. It matters only if the pilot passes its kill rule.

## Question

The README lists "mode routing combined with looping" as open. Looping means running one
block of layers several times per token instead of a deeper stack. This note says what that
combination could look like for a Qwen3 conversion, what is already published, and the one
experiment worth running first.

## Prior art (checked 2026-11-26)

- Relaxed Recursive Transformers (Bae et al., arXiv 2410.20672, ICLR 2025): converts a
  pretrained LLM into a smaller looped block, adds per-depth LoRA to relax the tying, and
  proposes continuous depth-wise batching with early exit.
- Mixture-of-Recursions (Bae et al., arXiv 2507.10524, NeurIPS 2025): a learned per-token
  router picks how many times the shared block runs. Two KV schemes: recursion-wise caching
  (only tokens that reach depth r store keys and values there) and recursive KV sharing (keys
  and values from the first pass reused at every depth).
- MoEUT (Csordás et al., arXiv 2405.16039, NeurIPS 2024): per-token MoE in both attention and
  MLP inside a looped layer group.
- From the project's earlier sweep, not re-checked tonight: Library of Layers (2611.03526,
  whole blocks per 128-token window) and per-loop-index LoRA (2610.40325).

None of these route by a discrete mode the model writes itself. Depth is either fixed,
chosen by a learned router, or chosen per window.

## Three ways to combine them

1. **Mode sets depth.** Think tokens run the looped block more times than answer or code
   tokens. No router: the mode tracker already gives the depth, at training and at
   generation, so training and inference agree exactly. The cost of a reasoning trace moves
   to the tokens that are thinking.
2. **Mode-by-loop adapters.** One shared looped MLP plus a LoRA per (mode, loop index), the
   relaxed-recursion recipe with a mode axis. Parameters grow with modes times loops times
   rank, so it only makes sense with LoRA modes.
3. **Mode-specific ends, shared looped middle.** Keep per-mode MLPs only in the first and
   last few layers and loop a shared middle block. Tests whether modes need the whole depth
   or only the edges; the probe results (tools/mode_probe.py) say at which layers the modes
   are already linearly separable, which is where specialisation could start.

## The KV cache is the hard part

The current design keeps one KV cache because attention is shared. With mode-dependent depth,
a token that ran 3 passes needs keys and values at pass 3 from earlier tokens that may have
stopped at pass 2 (an answer token before a new code or think segment). Options, cheapest
first:

- Share keys and values across passes (compute them in the first pass, reuse them in later
  passes, as in recursive KV sharing). The cache stays exactly the size it is now and a mode
  switch still never re-encodes anything. Cost: later passes attend only to first-pass keys.
- Fall back to a token's deepest available pass when a deeper one is missing. The cache grows
  by the extra passes of think tokens only.
- Re-run shallow tokens to the needed depth on a switch. Correct but brings back the
  re-prefill cost the design exists to avoid, so it is the baseline, not the plan.

## First experiment, if the pilot passes

No conversion to a looped model at first. Take the pilot's best mode model and, for think
tokens only, run a middle span of layers (for example layers 10 to 17) a second time with
keys and values shared across the two passes. Answer and code tokens are unchanged. Train
briefly with the same data and compare against the same model without the repeat:

- MATH500 accuracy and think-segment length at equal answer cost;
- whether answer quality moves at all (it should not, since answer tokens run unchanged);
- step time and memory, since think tokens are about 95% of training tokens and the repeat
  adds about 8/28 of the layer compute for them.

Kill rule: drop the looping line if the repeat does not improve MATH500 by more than seed
noise at two seeds, or if shared keys and values cost more than the repeat gains.

The variant with no shared keys and values (option 2 of the KV list) is the follow-up only if
the shared version shows a gain worth keeping.

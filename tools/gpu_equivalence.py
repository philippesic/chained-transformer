from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, get_args

import torch

from mode_mlp.generate import TokenTexts, mode_tracking
from mode_mlp.model import load_base, load_tokenizer
from mode_mlp.modes import MODE_NAMES
from mode_mlp.surgery import SurgeryConfig, Variant, apply_surgery

PROMPTS: list[tuple[str, bool]] = [
    ("Write a Python function that reverses a string, then explain it in one sentence.", False),
    ("Show a bash one-liner that counts lines in all .py files, in a code block.", False),
    ("What is 17 * 23? Then give a Python snippet that checks it.", True),
]


def switches(trace: list[int]) -> list[str]:
    out = []
    for a, b in zip(trace, trace[1:], strict=False):
        if a != b:
            out.append(f"{MODE_NAMES[a]}->{MODE_NAMES[b]}")
    return out


def run(variant: Variant, max_new_tokens: int, dtype: torch.dtype, device: str) -> dict[str, Any]:
    tok = load_tokenizer()
    texts = TokenTexts.from_tokenizer(tok)
    base = load_base(dtype=dtype, device=device)
    base.eval()
    routed = copy.deepcopy(base)
    routing = apply_surgery(routed, SurgeryConfig(variant=variant))
    routed.eval()
    results = []
    for prompt, thinking in PROMPTS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        with torch.no_grad():
            t0 = time.perf_counter()
            want = base.generate(ids, **gen_kwargs)
            base_s = time.perf_counter() - t0
            with mode_tracking(routed, routing, texts) as hook:
                t0 = time.perf_counter()
                got = routed.generate(ids, **gen_kwargs)
                routed_s = time.perf_counter() - t0
        a = want.sequences[0, ids.shape[1] :].tolist()
        b = got.sequences[0, ids.shape[1] :].tolist()
        first_diff = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
        shared = min(len(want.scores), len(got.scores))
        if first_diff is not None:
            shared = min(shared, first_diff + 1)
        logit_diff = max(float((want.scores[i] - got.scores[i]).abs().max()) for i in range(shared))
        trace = hook.trace[0]
        results.append(
            {
                "prompt": prompt,
                "enable_thinking": thinking,
                "new_tokens": len(b),
                "identical_tokens": a == b,
                "first_divergence": first_diff,
                "max_abs_logit_diff": logit_diff,
                "mode_switches": switches(trace[ids.shape[1] - 1 :]),
                "generated_tokens_by_mode": {
                    MODE_NAMES[m]: trace[ids.shape[1] - 1 :].count(m) for m in range(3)
                },
                "base_tok_per_s": round(len(a) / base_s, 1),
                "routed_tok_per_s": round(len(b) / routed_s, 1),
                "sample_tail": tok.decode(b[-60:]),
            }
        )
    return {
        "variant": variant,
        "dtype": str(dtype),
        "max_new_tokens": max_new_tokens,
        "device": device,
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if device == "cuda"
        else None,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Identical-mode generation equals the base model")
    parser.add_argument(
        "--variants", nargs="+", choices=get_args(Variant), default=list(get_args(Variant))
    )
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = [
        run(v, args.max_new_tokens, getattr(torch, args.dtype), args.device) for v in args.variants
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for r in report:
        for item in r["results"]:
            print(
                r["variant"],
                r["dtype"],
                item["identical_tokens"],
                item["first_divergence"],
                f"{item['max_abs_logit_diff']:.3g}",
                item["mode_switches"][:6],
            )


if __name__ == "__main__":
    main()

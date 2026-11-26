from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import GenerationConfig, Qwen3ForCausalLM

from chained_transformer.generate import TokenTexts, generate_separate_models, mode_tracking
from chained_transformer.modes import Mode
from chained_transformer.surgery import ModeMLP, Routing, SurgeryConfig, apply_surgery
from serving.mixed import batched_greedy_decode, use_grouped_mlps
from serving.tiny import TINY_PAD, TINY_PROMPTS, tiny_chained_model, tiny_texts

QWEN_PROMPTS: list[tuple[str, bool]] = [
    ("Write a Python function that reverses a string, then explain it in one sentence.", True),
    ("Show a bash one-liner that counts lines in all .py files, in a code block.", True),
    ("What is 17 * 23? Then give a Python snippet that checks it.", True),
    ("Explain what a hash map is, with a short Python example.", True),
]


def split_modes(model: Qwen3ForCausalLM) -> list[Qwen3ForCausalLM]:
    first = model.model.layers[0].mlp
    assert isinstance(first, ModeMLP)
    out = []
    for m in range(len(first.modes)):
        single = copy.deepcopy(model)
        for layer in single.model.layers:
            mlp = layer.mlp
            assert isinstance(mlp, ModeMLP)
            layer.mlp = mlp.modes[m]
        out.append(single)
    return out


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed(device: str, fn: Callable[[], Any]) -> tuple[Any, float]:
    sync(device)
    t0 = time.perf_counter()
    result = fn()
    sync(device)
    return result, time.perf_counter() - t0


def plain_greedy(
    model: nn.Module, prompts: Sequence[Sequence[int]], n: int, pad: int, device: str
) -> list[list[int]]:
    width = max(len(p) for p in prompts)
    ids = torch.tensor([[pad] * (width - len(p)) + list(p) for p in prompts], device=device)
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    cfg = GenerationConfig(max_new_tokens=n, do_sample=False, eos_token_id=None, pad_token_id=pad)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=mask, generation_config=cfg)  # type: ignore[operator]
    return [row[width:] for row in out.tolist()]


def setup(args: argparse.Namespace) -> tuple[Qwen3ForCausalLM, Routing, TokenTexts, list, int]:
    if args.model == "tiny":
        model, routing = tiny_chained_model()
        return model.to(args.device), routing, tiny_texts(), TINY_PROMPTS, TINY_PAD
    from chained_transformer.model import load_base, load_tokenizer

    tok = load_tokenizer(args.base)
    model = load_base(args.base, dtype=getattr(torch, args.dtype), device=args.device).eval()
    routing = apply_surgery(model, SurgeryConfig(variant="full"))
    if args.perturb > 0:
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".mlp.modes." in name and ".modes.0." not in name:
                    noise = torch.randn(p.shape, generator=g) * args.perturb
                    p.add_(noise.to(p.device, p.dtype) * p.abs().mean())
    prompts = []
    for text, thinking in QWEN_PROMPTS[: args.prompts]:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        prompts.append(tok(chat, add_special_tokens=False).input_ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return model, routing, TokenTexts.from_tokenizer(tok), prompts, int(pad)


def arm(name: str, seconds: float, tokens: int, **extra: Any) -> dict[str, Any]:
    row = {"arm": name, "seconds": round(seconds, 4), "new_tokens": tokens}
    row["tok_per_s"] = round(tokens / seconds, 1) if seconds > 0 else None
    row.update(extra)
    print(f"{name:24s} {seconds:8.3f}s {tokens:6d} tok {row['tok_per_s']} tok/s {extra}")
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    model, routing, texts, prompts, pad = setup(args)
    n = args.max_new_tokens
    device = args.device
    separate = split_modes(model)
    base = separate[0]
    rows = []

    def base_each() -> list[list[int]]:
        return [plain_greedy(base, [p], n, pad, device)[0] for p in prompts]

    out, s = timed(device, base_each)
    rows.append(arm("base_per_sequence", s, sum(map(len, out))))
    out, s = timed(device, lambda: plain_greedy(base, prompts, n, pad, device))
    rows.append(arm("base_batched", s, sum(map(len, out))))

    def chained_each() -> tuple[list[list[int]], list[list[int]]]:
        tokens, traces = [], []
        cfg = GenerationConfig(
            max_new_tokens=n, do_sample=False, eos_token_id=None, pad_token_id=pad
        )
        for p in prompts:
            with torch.no_grad(), mode_tracking(model, routing, texts) as hook:
                o = model.generate(torch.tensor([p], device=device), generation_config=cfg)
            tokens.append(o[0, len(p) :].tolist())
            traces.append(hook.trace[0])
        return tokens, traces

    (chained_tokens, traces), s = timed(device, chained_each)
    switches = sum(
        sum(a != b for a, b in zip(t[len(p) - 1 :], t[len(p) :], strict=False))
        for t, p in zip(traces, prompts, strict=True)
    )
    rows.append(
        arm("chained_per_sequence", s, sum(map(len, chained_tokens)), mode_switches=switches)
    )

    grouped = copy.deepcopy(model)
    grouped_routing = Routing()
    use_grouped_mlps(grouped, grouped_routing, args.grouped_impl)
    with torch.no_grad():
        batched, s = timed(
            device,
            lambda: batched_greedy_decode(grouped, grouped_routing, texts, prompts, n, pad),
        )
    rows.append(
        arm(
            "chained_batched",
            s,
            sum(map(len, batched.token_ids)),
            mixed_steps=batched.mixed_steps,
            matches_per_sequence=batched.token_ids == chained_tokens,
        )
    )

    def separate_each() -> list[Any]:
        return [
            generate_separate_models(separate, texts, p, n, eos_ids=set(), prompt_mode=Mode.THINK)
            for p in prompts
        ]

    results, s = timed(device, separate_each)
    rows.append(
        arm(
            "separate_reprefill",
            s,
            sum(len(r.token_ids) for r in results),
            mode_switches=sum(r.switches for r in results),
            reprefilled_tokens=sum(r.reprefilled_tokens for r in results),
            prompt_tokens=sum(map(len, prompts)),
        )
    )
    return {
        "model": args.model if args.model == "tiny" else args.base,
        "device": device,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "prompts": len(prompts),
        "max_new_tokens": n,
        "perturb": args.perturb,
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if device == "cuda"
        else None,
        "arms": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Base model vs chained modes vs separate fine-tunes with re-prefill (HF)"
    )
    parser.add_argument("--model", choices=["tiny", "qwen"], default="tiny")
    parser.add_argument("--base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--prompts", type=int, default=len(QWEN_PROMPTS))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--perturb", type=float, default=0.0)
    parser.add_argument("--grouped-impl", default="grouped")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

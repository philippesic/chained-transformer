from __future__ import annotations

import argparse
import copy
import gc
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from chained_transformer.checkpoint import (
    VARIANTS,
    convert,
    load_checkpoint,
    meta_counts,
    save_checkpoint,
    shape_config,
)
from chained_transformer.model import load_base
from chained_transformer.surgery import Routing, SurgeryConfig, Variant


def schedules(batch: int, seq: int) -> dict[str, torch.Tensor]:
    switching = torch.zeros(batch, seq, dtype=torch.long)
    third = seq // 3
    switching[:, third : 2 * third] = 1
    switching[:, 2 * third :] = 2
    g = torch.Generator().manual_seed(0)
    return {
        "think": torch.zeros(batch, seq, dtype=torch.long),
        "code": torch.ones(batch, seq, dtype=torch.long),
        "answer": torch.full((batch, seq), 2, dtype=torch.long),
        "think_code_answer": switching,
        "random": torch.randint(0, 3, (batch, seq), generator=g),
    }


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    return {"bitwise": bool(torch.equal(a, b)), "max_abs_diff": float((a - b).abs().max())}


@torch.no_grad()
def prefill_diffs(
    base: Qwen3ForCausalLM, model: Qwen3ForCausalLM, routing: Routing, ids: torch.Tensor
) -> dict[str, Any]:
    want = base(input_ids=ids).logits
    out = {}
    for name, modes in schedules(*ids.shape).items():
        routing.set(modes)
        out[name] = compare(model(input_ids=ids).logits, want)
    return out


@torch.no_grad()
def decode_diff(
    base: Qwen3ForCausalLM, model: Qwen3ForCausalLM, routing: Routing, ids: torch.Tensor
) -> dict[str, Any]:
    modes = schedules(1, ids.shape[1])["think_code_answer"]
    prompt = ids.shape[1] // 2
    want_cache, got_cache = DynamicCache(), DynamicCache()
    want = [base(input_ids=ids[:, :prompt], cache=want_cache, use_cache=True).logits]
    routing.set(modes[:, :prompt])
    got = [model(input_ids=ids[:, :prompt], cache=got_cache, use_cache=True).logits]
    for t in range(prompt, ids.shape[1]):
        step = ids[:, t : t + 1]
        want.append(base(input_ids=step, cache=want_cache, use_cache=True).logits)
        routing.set(modes[:, t : t + 1])
        got.append(model(input_ids=step, cache=got_cache, use_cache=True).logits)
    return compare(torch.cat(got, dim=1), torch.cat(want, dim=1))


def make_base(args: argparse.Namespace) -> Qwen3ForCausalLM:
    if args.random_shape:
        torch.manual_seed(args.seed)
        model = Qwen3ForCausalLM(shape_config(args.base)).float()
    else:
        model = load_base(args.base, dtype=torch.float32, device="cpu")
    return model.eval()


def check_variant(
    base: Qwen3ForCausalLM, variant: Variant, ids: torch.Tensor, save_dir: Path | None
) -> dict[str, Any]:
    start = time.perf_counter()
    model = convert(copy.deepcopy(base), SurgeryConfig(variant=variant), "check").eval()
    result: dict[str, Any] = {
        "convert_s": round(time.perf_counter() - start, 2),
        "params": meta_counts(model.config),
        "prefill": prefill_diffs(base, model, model.routing, ids),
        "decode_switching": decode_diff(base, model, model.routing, ids[:1]),
    }
    if save_dir is not None:
        target = save_dir / variant
        start = time.perf_counter()
        save_checkpoint(model, target)
        loaded = load_checkpoint(target, dtype=torch.float32)
        assert loaded.routing is not None
        saved = model.state_dict()
        reread = loaded.model.state_dict()
        result["round_trip"] = {
            "seconds": round(time.perf_counter() - start, 2),
            "same_keys": saved.keys() == reread.keys(),
            "all_bitwise": all(torch.equal(saved[k], reread[k]) for k in saved),
            "tied": loaded.model.lm_head.weight is loaded.model.model.embed_tokens.weight,
            "bytes": sum(p.stat().st_size for p in target.iterdir()),
        }
        del loaded
    del model
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that fresh conversions reproduce the base")
    parser.add_argument("--base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--random-shape", action="store_true", help="random weights, real shapes")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round-trip", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    base = make_base(args)
    g = torch.Generator().manual_seed(args.seed + 1)
    ids = torch.randint(0, base.config.vocab_size, (args.batch, args.seq), generator=g)
    report: dict[str, Any] = {
        "base": args.base,
        "weights": "random (real shapes)" if args.random_shape else "hub",
        "dtype": "float32",
        "device": "cpu",
        "tokens": list(ids.shape),
        "base_params": sum(p.numel() for p in base.parameters()),
        "variants": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp) if args.round_trip else None
        for variant in (v for v in VARIANTS if v in args.variants):
            report["variants"][variant] = check_variant(base, variant, ids, save_dir)
            print(variant, json.dumps(report["variants"][variant]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

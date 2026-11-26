from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from chained_transformer.modes import Mode, ModeTracker

ARMS = ("base", "chained", "separate")


def trained_model(path: Path) -> Any:
    import torch

    from chained_transformer.checkpoint import load_checkpoint

    built = load_checkpoint(path, dtype=torch.float32)
    if built.routing is None:
        raise SystemExit(f"{path} is a plain Qwen3 model, not a mode checkpoint")
    variant = built.model.config.mode_surgery["variant"]
    if variant not in ("full", "lora"):
        raise SystemExit(
            f"{path} is a {variant!r} checkpoint; vLLM serving covers per-mode MLPs "
            "(full and lora), and the moe arm routes by a learned router instead"
        )
    return built.model.eval()


def prepare(args: argparse.Namespace) -> None:
    import torch

    from chained_transformer.surgery import SurgeryConfig, apply_surgery
    from serving.export import export_for_vllm, export_single_mode

    out: Path = args.dir
    dtype = getattr(torch, args.dtype)
    if args.model == "tiny":
        from serving.tiny import TINY_PROMPTS, tiny_chained_model, tiny_token_texts

        model = trained_model(args.checkpoint) if args.checkpoint else tiny_chained_model()[0]
        export_for_vllm(model, out / "chained", token_texts=tiny_token_texts(), dtype=dtype)
        prompts = [TINY_PROMPTS[i % len(TINY_PROMPTS)] for i in range(args.num_prompts)]
        texts = tiny_token_texts()
    else:
        from chained_transformer.generate import TokenTexts
        from chained_transformer.model import load_base, load_tokenizer
        from tools.bench_serving import QWEN_PROMPTS

        if args.checkpoint is not None:
            model = trained_model(args.checkpoint)
            has_tokenizer = (args.checkpoint / "tokenizer_config.json").exists()
            tok = load_tokenizer(str(args.checkpoint) if has_tokenizer else args.base)
        else:
            tok = load_tokenizer(args.base)
            model = load_base(args.base, dtype=torch.float32, device="cpu").eval()
            apply_surgery(model, SurgeryConfig(variant="full"))
        if args.perturb > 0:
            g = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if ".mlp.modes." in name and ".modes.0." not in name:
                        p.add_(torch.randn(p.shape, generator=g) * args.perturb * p.abs().mean())
        export_for_vllm(model, out / "chained", dtype=dtype)
        tok.save_pretrained(out / "chained")
        prompts = []
        for i in range(args.num_prompts):
            text, thinking = QWEN_PROMPTS[i % len(QWEN_PROMPTS)]
            chat = tok.apply_chat_template(
                [{"role": "user", "content": f"Request {i}. {text}"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            prompts.append(tok(chat, add_special_tokens=False).input_ids)
        texts = TokenTexts.from_tokenizer(tok).texts
    for m in range(len(Mode)):
        export_single_mode(model, m, out / f"mode{m}", dtype=dtype)
        if args.model != "tiny":
            tok.save_pretrained(out / f"mode{m}")
    (out / "prompts.json").write_text(json.dumps(prompts) + "\n")
    (out / "texts.json").write_text(json.dumps(texts) + "\n")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "base": args.base,
                "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
                "dtype": args.dtype,
            }
        )
        + "\n"
    )
    print(f"prepared {out}")


def engine(path: Path, args: argparse.Namespace, tiny: bool, **extra: Any) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": str(path),
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": True,
        "seed": 0,
        "gpu_memory_utilization": args.gpu_mem,
        "enforce_eager": args.eager,
    }
    if tiny:
        kwargs.update(skip_tokenizer_init=True, block_size=32)
    kwargs.update(extra)
    return LLM(**kwargs)


def sampling(max_tokens: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True, detokenize=False)


def prefill_computed(prompt_len: int, out: Any) -> int:
    return prompt_len - int(getattr(out, "num_cached_tokens", 0) or 0)


def one_shot(llm: Any, prompts: list[list[int]], max_tokens: int) -> dict[str, Any]:
    from vllm.inputs import TokensPrompt

    llm.generate([TokensPrompt(prompt_token_ids=prompts[0])], sampling(4))
    t0 = time.perf_counter()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sampling(max_tokens))
    seconds = time.perf_counter() - t0
    return {
        "seconds": seconds,
        "new_tokens": sum(len(o.outputs[0].token_ids) for o in outs),
        "prefill_tokens": sum(
            prefill_computed(len(p), o) for p, o in zip(prompts, outs, strict=True)
        ),
        "calls": 1,
        "outputs": [list(o.outputs[0].token_ids) for o in outs],
    }


def separate(
    llms: list[Any], prompts: list[list[int]], texts: list[str], max_tokens: int, segment: int
) -> dict[str, Any]:
    from vllm.inputs import TokensPrompt

    def text(t: int) -> str:
        return texts[t] if t < len(texts) else ""

    for llm in llms:
        llm.generate([TokensPrompt(prompt_token_ids=prompts[0])], sampling(4))
    tokens = [list(p) for p in prompts]
    trackers = [ModeTracker(prompt_mode=Mode.THINK) for _ in prompts]
    modes = [
        int([tr.feed(text(t)) for t in p][-1]) for tr, p in zip(trackers, prompts, strict=True)
    ]
    new = [0] * len(prompts)
    stats = {"calls": 0, "prefill_tokens": 0, "wasted_tokens": 0, "switches": 0}
    t0 = time.perf_counter()
    while any(n < max_tokens for n in new):
        for m, llm in enumerate(llms):
            batch = [i for i in range(len(prompts)) if new[i] < max_tokens and modes[i] == m]
            if not batch:
                continue
            outs = llm.generate(
                [TokensPrompt(prompt_token_ids=tokens[i]) for i in batch],
                [sampling(min(segment, max_tokens - new[i])) for i in batch],
            )
            stats["calls"] += 1
            for i, out in zip(batch, outs, strict=True):
                stats["prefill_tokens"] += prefill_computed(len(tokens[i]), out)
                gen = list(out.outputs[0].token_ids)
                kept = 0
                for t in gen:
                    kept += 1
                    mode = int(trackers[i].feed(text(t)))
                    if mode != modes[i]:
                        modes[i] = mode
                        stats["switches"] += 1
                        break
                stats["wasted_tokens"] += len(gen) - kept
                tokens[i] += gen[:kept]
                new[i] += kept
    seconds = time.perf_counter() - t0
    return {
        "seconds": seconds,
        "new_tokens": sum(new),
        **stats,
        "outputs": [t[len(p) :] for t, p in zip(tokens, prompts, strict=True)],
    }


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    meta = json.loads((args.dir / "meta.json").read_text())
    tiny = meta["model"] == "tiny"
    prompts = json.loads((args.dir / "prompts.json").read_text())
    texts = json.loads((args.dir / "texts.json").read_text())
    chained = args.arm == "chained"
    row: dict[str, Any] = {
        "arm": args.arm,
        "impl": args.impl if chained else None,
        "eager": args.eager or (chained and args.impl == "loop"),
    }
    if args.arm == "base":
        row.update(one_shot(engine(args.dir / "mode0", args, tiny), prompts, args.max_tokens))
    elif args.arm == "chained":
        from serving.vllm_plugin import register
        from serving.vllm_plugin.state import collect_traces

        register()
        tracking: dict[str, Any] = {"prompt_mode": int(Mode.THINK), "mlp_impl": args.impl}
        if tiny:
            tracking["token_texts_file"] = "token_texts.json"
        llm = engine(
            args.dir / "chained",
            args,
            tiny,
            hf_overrides={"mode_tracking": tracking},
            enforce_eager=args.eager or args.impl == "loop",
        )
        row.update(one_shot(llm, prompts, args.max_tokens))
        stats = llm.collective_rpc(collect_traces)[0]
        row.update(steps=stats["steps"], mixed_steps=stats["mixed_steps"])
    else:
        share = args.gpu_mem / len(Mode)
        llms = [
            engine(args.dir / f"mode{m}", args, tiny, gpu_memory_utilization=share)
            for m in range(len(Mode))
        ]
        row.update(separate(llms, prompts, texts, args.max_tokens, args.segment))
    row["tok_per_s"] = round(row["new_tokens"] / row["seconds"], 1)
    return row


def run_all(args: argparse.Namespace) -> None:
    if not (args.dir / "meta.json").exists():
        prepare(args)
    rows = []
    plan = [("base", None), *(("chained", impl) for impl in args.impls), ("separate", None)]
    for arm, impl in plan:
        out = args.dir / f"row_{arm}_{impl or 'na'}.json"
        cmd = [sys.executable, "-m", "tools.bench_vllm", "arm", "--dir", str(args.dir)]
        cmd += ["--arm", arm, "--impl", impl or "select", "--json-out", str(out)]
        cmd += ["--dtype", args.dtype, "--max-tokens", str(args.max_tokens)]
        cmd += ["--segment", str(args.segment), "--gpu-mem", str(args.gpu_mem)]
        cmd += ["--max-model-len", str(args.max_model_len)]
        if args.eager:
            cmd.append("--eager")
        env = dict(os.environ)
        if arm != "separate":
            env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        print("+", " ".join(cmd), flush=True)
        done = subprocess.run(cmd, env=env)
        if done.returncode != 0 or not out.exists():
            rows.append({"arm": arm, "impl": impl, "error": f"exit {done.returncode}"})
            continue
        rows.append(json.loads(out.read_text()))
    base_out = next((r.get("outputs") for r in rows if r["arm"] == "base"), None)
    for r in rows:
        outputs = r.pop("outputs", None)
        if outputs is not None and base_out is not None:
            r["same_tokens_as_base"] = sum(a == b for a, b in zip(outputs, base_out, strict=True))
        print({k: v for k, v in r.items()})
    report = {"dir": str(args.dir), "max_tokens": args.max_tokens, "arms": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM: base vs chained modes vs separate models")
    parser.add_argument("command", choices=["prepare", "arm", "all"])
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--model", choices=["tiny", "qwen"], default="qwen")
    parser.add_argument("--base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--perturb", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--arm", choices=ARMS, default="base")
    parser.add_argument("--impl", default="select")
    parser.add_argument("--impls", nargs="+", default=["select", "triton"])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--segment", type=int, default=16)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.checkpoint is not None and args.perturb > 0:
        parser.error("--perturb adds noise to copies of the base MLP; a --checkpoint has its own")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "arm":
        row = run_arm(args)
        assert args.json_out is not None
        args.json_out.write_text(json.dumps(row) + "\n")
    else:
        assert args.out is not None
        run_all(args)


if __name__ == "__main__":
    main()

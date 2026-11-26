from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig, PreTrainedTokenizerBase
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from mode_mlp.generate import TokenTexts, mode_tracking
from mode_mlp.modes import MODE_NAMES, THINK_CLOSE, Mode, ModeTracker
from mode_mlp.surgery import Routing

MATH500 = "HuggingFaceH4/MATH-500"
LIVECODEBENCH = "livecodebench/code_generation_lite"
THINK_WORDS = ("wait", "hmm", "let me", "alternatively", "actually", "double-check", "i think")
THINK_WORD_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in THINK_WORDS) + r")\b", re.I)


@dataclass(frozen=True)
class Problem:
    task_id: str
    prompt: str
    reference: str


@dataclass(frozen=True)
class Response:
    task_id: str
    text: str
    new_tokens: int
    tokens_by_mode: dict[str, int]


def segment_text(response: str) -> dict[str, str]:
    tracker = ModeTracker()
    tracker.feed("<|im_start|>assistant\n")
    parts: dict[str, list[str]] = {name: [] for name in MODE_NAMES}
    for ch in response:
        parts[MODE_NAMES[tracker.mode]].append(ch)
        tracker.feed(ch)
    return {name: "".join(chars) for name, chars in parts.items()}


def final_answer(response: str) -> str:
    return segment_text(response)["answer"] or response


def after_thinking(response: str) -> str:
    head, sep, tail = response.partition(THINK_CLOSE)
    return tail if sep else head


def think_word_leakage(response: str) -> float:
    answer = segment_text(response)["answer"]
    words = len(answer.split())
    return 1000.0 * len(THINK_WORD_RE.findall(answer)) / words if words else 0.0


def score_math(response: str, reference: str) -> bool:
    from math_verify import parse, verify

    gold = parse(f"${reference}$")
    return bool(verify(gold, parse(final_answer(response))))


def load_math500(limit: int | None) -> list[Problem]:
    from datasets import load_dataset

    rows = load_dataset(MATH500, split="test")
    picked = rows.select(range(min(limit, len(rows)))) if limit else rows
    suffix = "\n\nPut the final answer in \\boxed{}."
    return [Problem(r["unique_id"], r["problem"] + suffix, r["answer"]) for r in picked]


def load_humanevalplus(limit: int | None) -> list[Problem]:
    from evalplus.data import get_human_eval_plus

    problems = list(get_human_eval_plus().values())[:limit]
    lead = "Complete this Python function. Reply with one ```python code block.\n\n"
    return [Problem(p["task_id"], lead + p["prompt"], p["entry_point"]) for p in problems]


def extract_code(response: str) -> str:
    visible = after_thinking(response)
    blocks = re.findall(r"```(?:python)?\n(.*?)```", visible, re.S)
    return blocks[-1] if blocks else visible


def write_evalplus_samples(responses: Sequence[Response], path: Path) -> None:
    with path.open("w") as f:
        for r in responses:
            f.write(json.dumps({"task_id": r.task_id, "solution": extract_code(r.text)}) + "\n")


def run_evalplus(samples: Path) -> None:
    subprocess.run(
        ["evalplus.evaluate", "--dataset", "humaneval", "--samples", str(samples)], check=True
    )


def load_livecodebench(limit: int | None) -> list[Problem]:
    raise NotImplementedError(
        f"LiveCodeBench scoring needs the lcb_runner harness, which is not on PyPI; "
        f"pin a commit of it before adding {LIVECODEBENCH} problems here"
    )


@torch.no_grad()
def generate_responses(
    model: Qwen3ForCausalLM,
    routing: Routing | None,
    tokenizer: PreTrainedTokenizerBase,
    problems: Sequence[Problem],
    max_new_tokens: int,
) -> tuple[list[Response], dict[str, float]]:
    texts = TokenTexts.from_tokenizer(tokenizer)
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]),
        pad_token_id=tokenizer.pad_token_id,
    )
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    responses = []
    generated = 0
    start = time.perf_counter()
    for problem in problems:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        ids = ids.to(device)
        if routing is None:
            out = model.generate(ids, generation_config=config)
            new = out[0, ids.shape[1] :].tolist()
            modes = texts.modes(out[0].tolist())[ids.shape[1] - 1 : -1]
        else:
            with mode_tracking(model, routing, texts) as hook:
                out = model.generate(ids, generation_config=config)
            new = out[0, ids.shape[1] :].tolist()
            modes = hook.trace[0][ids.shape[1] - 1 :]
        generated += len(new)
        responses.append(
            Response(
                problem.task_id,
                str(tokenizer.decode(new, skip_special_tokens=True)),
                len(new),
                {MODE_NAMES[m]: modes.count(int(m)) for m in Mode},
            )
        )
    elapsed = time.perf_counter() - start
    perf = {
        "tokens_per_s": generated / elapsed,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    return responses, perf


def summarize(
    name: str, problems: Sequence[Problem], responses: Sequence[Response], perf: dict[str, float]
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "benchmark": name,
        "n": len(responses),
        "think_word_leakage_per_1k": sum(think_word_leakage(r.text) for r in responses)
        / max(1, len(responses)),
        **perf,
    }
    if name == "math500":
        refs = {p.task_id: p.reference for p in problems}
        correct = sum(score_math(r.text, refs[r.task_id]) for r in responses)
        summary["accuracy"] = correct / max(1, len(responses))
    return summary


def main() -> None:
    from mode_mlp.model import load_base, load_tokenizer

    parser = argparse.ArgumentParser(description="Stub eval runner (base model only for now)")
    parser.add_argument("benchmark", choices=["math500", "humanevalplus", "livecodebench"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    loaders = {
        "math500": load_math500,
        "humanevalplus": load_humanevalplus,
        "livecodebench": load_livecodebench,
    }
    problems = loaders[args.benchmark](args.limit)
    tokenizer = load_tokenizer()
    model = load_base().eval()
    responses, perf = generate_responses(model, None, tokenizer, problems, args.max_new_tokens)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "responses.jsonl").open("w") as f:
        for r in responses:
            f.write(json.dumps(r.__dict__) + "\n")
    if args.benchmark == "humanevalplus":
        write_evalplus_samples(responses, args.out / "samples.jsonl")
        run_evalplus(args.out / "samples.jsonl")
    summary = summarize(args.benchmark, problems, responses, perf)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

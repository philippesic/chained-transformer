from __future__ import annotations

import resource
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from chained_transformer.checkpoint import load_checkpoint
from chained_transformer.generate import TokenTexts, generate_separate_models, mode_tracking
from chained_transformer.modes import MODE_NAMES, Mode
from chained_transformer.surgery import Routing
from evals.types import Problem

DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
EOS_TOKENS = ("<|im_end|>", "<|endoftext|>")


@dataclass(frozen=True)
class GenSettings:
    max_new_tokens: int = 32768
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    batch_size: int = 1
    seed: int = 0
    enable_thinking: bool = True


@dataclass(frozen=True)
class Generation:
    task_id: str
    text: str
    new_tokens: int
    finished: bool
    tokens_by_mode: dict[str, int]
    switches: int
    reprefilled_tokens: int = 0


@dataclass
class Runner:
    models: list[Qwen3ForCausalLM]
    routing: Routing | None
    tokenizer: PreTrainedTokenizerBase
    spec: dict[str, Any] = field(default_factory=dict)

    @property
    def separate(self) -> bool:
        return len(self.models) > 1

    @property
    def device(self) -> torch.device:
        return next(self.models[0].parameters()).device


def _has_tokenizer(path: str) -> bool:
    return (Path(path) / "tokenizer_config.json").exists()


def describe(model: Qwen3ForCausalLM, path: str) -> dict[str, Any]:
    config = model.config
    surgery = getattr(config, "mode_surgery", None)
    return {
        "path": path,
        "model_type": config.model_type,
        "variant": surgery["variant"] if surgery else None,
        "mode_surgery": surgery,
        "base_model": getattr(config, "base_model", None),
        "params": sum(p.numel() for p in model.parameters()),
    }


def load_runner(
    model: str | None,
    separate: Sequence[str] | None,
    tokenizer: str | None,
    dtype: torch.dtype,
    device: str,
) -> Runner:
    paths = list(separate) if separate else [model] if model else []
    if len(paths) not in (1, len(MODE_NAMES)):
        raise ValueError(f"give one model or {len(MODE_NAMES)} separate models (think code answer)")
    built = [load_checkpoint(p, dtype=dtype, device=device) for p in paths]
    if len(built) > 1 and any(b.routing is not None for b in built):
        raise ValueError("the separate arm takes plain fine-tunes, not mode checkpoints")
    tok_name = tokenizer or (paths[0] if _has_tokenizer(paths[0]) else DEFAULT_TOKENIZER)
    tok = AutoTokenizer.from_pretrained(tok_name)
    tok.padding_side = "left"
    for b in built:
        b.model.eval()
    spec = {
        "arm": "separate" if len(built) > 1 else (built[0].model.config.model_type),
        "models": [describe(b.model, p) for b, p in zip(built, paths, strict=True)],
        "tokenizer": tok_name,
        "dtype": str(dtype).removeprefix("torch."),
        "device": device,
    }
    return Runner([b.model for b in built], built[0].routing, tok, spec)


def chat_prompt(tokenizer: PreTrainedTokenizerBase, problem: Problem, enable_thinking: bool) -> str:
    messages = []
    system = problem.meta.get("system")
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": problem.prompt})
    return str(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    )


def labels_of(position_modes: Sequence[int]) -> dict[str, int]:
    return {MODE_NAMES[m]: sum(1 for x in position_modes if x == m) for m in Mode}


def switches_of(labels: Sequence[int]) -> int:
    return sum(1 for a, b in zip(labels, labels[1:], strict=False) if a != b)


def _trim(new: list[int], eos: set[int]) -> tuple[list[int], bool]:
    for i, t in enumerate(new):
        if t in eos:
            return new[: i + 1], True
    return new, False


def _eos_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    ids = [tokenizer.convert_tokens_to_ids(t) for t in EOS_TOKENS]
    return [i for i in ids if isinstance(i, int) and i != tokenizer.unk_token_id]


@torch.no_grad()
def _generate_batch(
    runner: Runner,
    texts: TokenTexts,
    problems: Sequence[Problem],
    settings: GenSettings,
    seed: int,
) -> list[Generation]:
    tok = runner.tokenizer
    eos = _eos_ids(tok)
    prompts = [chat_prompt(tok, p, settings.enable_thinking) for p in problems]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    ids = enc.input_ids.to(runner.device)
    mask = enc.attention_mask.to(runner.device)
    config = GenerationConfig(
        max_new_tokens=settings.max_new_tokens,
        do_sample=settings.do_sample,
        temperature=settings.temperature if settings.do_sample else None,
        top_p=settings.top_p if settings.do_sample else None,
        top_k=settings.top_k if settings.do_sample else None,
        eos_token_id=eos,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos[0],
    )
    torch.manual_seed(seed)
    model = runner.models[0]
    width = ids.shape[1]
    if runner.routing is None:
        out = model.generate(input_ids=ids, attention_mask=mask, generation_config=config)
        rows: list[list[int]] = out.tolist()
        traces = [texts.modes(row)[:-1] for row in rows]
    else:
        with mode_tracking(model, runner.routing, texts) as hook:
            out = model.generate(input_ids=ids, attention_mask=mask, generation_config=config)
        traces = hook.trace
    results = []
    for row, problem in enumerate(problems):
        new, finished = _trim(out[row, width:].tolist(), set(eos))
        labels = traces[row][width - 1 : width - 1 + len(new)]
        results.append(
            Generation(
                problem.task_id,
                str(tok.decode(new, skip_special_tokens=True)),
                len(new),
                finished,
                labels_of(labels),
                switches_of(labels),
            )
        )
    return results


def _generate_separate(
    runner: Runner, texts: TokenTexts, problem: Problem, settings: GenSettings
) -> Generation:
    if settings.do_sample:
        raise ValueError("the separate arm decodes greedily; pass --greedy")
    tok = runner.tokenizer
    eos = set(_eos_ids(tok))
    prompt = chat_prompt(tok, problem, settings.enable_thinking)
    ids = tok(prompt, add_special_tokens=False).input_ids
    res = generate_separate_models(runner.models, texts, ids, settings.max_new_tokens, eos)
    new, finished = _trim(res.token_ids, eos)
    labels = res.modes[len(ids) - 1 : len(ids) - 1 + len(new)]
    return Generation(
        problem.task_id,
        str(tok.decode(new, skip_special_tokens=True)),
        len(new),
        finished,
        labels_of(labels),
        switches_of(labels),
        res.reprefilled_tokens,
    )


def peak_memory_gib(device: torch.device) -> dict[str, float]:
    if device.type == "cuda":
        return {"peak_alloc_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3)}
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss_gib": round(rss_kib / 2**20, 3)}


def generate_all(
    runner: Runner,
    problems: Sequence[Problem],
    settings: GenSettings,
    on_done: Callable[[Generation], None] | None = None,
) -> tuple[list[Generation], dict[str, Any]]:
    texts = TokenTexts.from_tokenizer(runner.tokenizer)
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)
    results: list[Generation] = []
    start = time.perf_counter()
    step = 1 if runner.separate else max(1, settings.batch_size)
    for i in range(0, len(problems), step):
        chunk = problems[i : i + step]
        if runner.separate:
            batch = [_generate_separate(runner, texts, chunk[0], settings)]
        else:
            batch = _generate_batch(runner, texts, chunk, settings, settings.seed + i)
        for g in batch:
            results.append(g)
            if on_done is not None:
                on_done(g)
    elapsed = time.perf_counter() - start
    tokens = sum(g.new_tokens for g in results)
    perf = {
        "seconds": round(elapsed, 2),
        "new_tokens": tokens,
        "tokens_per_s": round(tokens / elapsed, 2) if elapsed > 0 else 0.0,
        **peak_memory_gib(runner.device),
    }
    return results, perf


def settings_dict(settings: GenSettings) -> dict[str, Any]:
    return asdict(settings)

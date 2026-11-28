from __future__ import annotations

import resource
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from chained_transformer.attention import FALLBACKS, use_attention
from chained_transformer.cache import roomy_cache
from chained_transformer.checkpoint import load_checkpoint
from chained_transformer.generate import TokenTexts, generate_separate_models, mode_tracking
from chained_transformer.graph_decode import greedy_generate
from chained_transformer.modes import MODE_NAMES, Mode
from chained_transformer.surgery import Routing
from evals.types import Problem

DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
EOS_TOKENS = ("<|im_end|>", "<|endoftext|>")
AUTO_BATCH_CAP = 16
KV_FRACTION = 0.8
REFILL_BELOW = 0.5
PREFILL_CHUNK = 512
MIN_CALL = 256
LOOP_TAIL = 768
LOOP_PERIOD = 256
LOOP_CHECK = 128
GiB = 2**30


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
    refill: bool = True
    stop_loops: bool = False
    decode: str = "hf"


@dataclass(frozen=True)
class Generation:
    task_id: str
    text: str
    new_tokens: int
    finished: bool
    tokens_by_mode: dict[str, int]
    switches: int
    reprefilled_tokens: int = 0
    mode_runs: list[list[int]] = field(default_factory=list)
    stopped: str | None = None


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
    attention: str = "auto",
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
        use_attention(b.model, attention)
    spec = {
        "arm": "separate" if len(built) > 1 else (built[0].model.config.model_type),
        "models": [describe(b.model, p) for b, p in zip(built, paths, strict=True)],
        "tokenizer": tok_name,
        "dtype": str(dtype).removeprefix("torch."),
        "device": device,
        "attention": attention,
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


def runs_of(labels: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for m in labels:
        if runs and runs[-1][0] == m:
            runs[-1][1] += 1
        else:
            runs.append([int(m), 1])
    return runs


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
                mode_runs=runs_of(labels),
            )
        )
    return results


@dataclass(eq=False)
class _Slot:
    problem: Problem
    prompt: list[int]
    new: list[int] = field(default_factory=list)
    order: int = 0

    @property
    def length(self) -> int:
        return len(self.prompt) + len(self.new)


def loop_period(tail: Sequence[int]) -> int | None:
    if len(tail) < LOOP_TAIL + LOOP_PERIOD:
        return None
    window = list(tail[-LOOP_TAIL:])
    for period in range(1, LOOP_PERIOD + 1):
        if window == list(tail[-LOOP_TAIL - period : -period]):
            return period
    return None


class _StopOrRefill(StoppingCriteria):
    def __init__(
        self,
        width: int,
        budgets: list[int],
        eos: list[int],
        low: int,
        device: torch.device,
        written: list[int] | None = None,
    ) -> None:
        self.width = width
        self.budgets = torch.tensor(budgets, device=device)
        self.eos = torch.tensor(eos, device=device)
        self.ended = torch.zeros(len(budgets), dtype=torch.bool, device=device)
        self.looped = torch.zeros(len(budgets), dtype=torch.bool, device=device)
        self.low = low
        self.early = False
        self.written = written
        self.loop_at: dict[int, int] = {}

    def _mark_loops(self, input_ids: torch.LongTensor, steps: int) -> None:
        assert self.written is not None
        span = LOOP_TAIL + LOOP_PERIOD
        live = ~(self.ended | self.looped | (self.budgets <= steps))
        rows = [i for i in live.nonzero().flatten().tolist() if self.written[i] + steps >= span]
        if not rows:
            return
        tails = input_ids[rows, -span:].tolist()
        for i, tail in zip(rows, tails, strict=True):
            if loop_period(tail) is not None:
                self.looped[i] = True
                self.loop_at[i] = steps

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> torch.BoolTensor:
        steps = input_ids.shape[1] - self.width
        self.ended |= torch.isin(input_ids[:, -1], self.eos)
        if self.written is not None and steps % LOOP_CHECK == 0:
            self._mark_loops(input_ids, steps)
        done = self.ended | self.looped | (self.budgets <= steps)
        if int((~done).sum()) <= self.low:
            self.early = True
            return cast(torch.BoolTensor, torch.ones_like(done))
        return cast(torch.BoolTensor, done)


def _finish(
    slot: _Slot,
    finished: bool,
    texts: TokenTexts,
    tok: PreTrainedTokenizerBase,
    stopped: str | None = None,
) -> Generation:
    start = len(slot.prompt) - 1
    labels = texts.modes(slot.prompt + slot.new)[start : start + len(slot.new)]
    return Generation(
        slot.problem.task_id,
        str(tok.decode(slot.new, skip_special_tokens=True)),
        len(slot.new),
        finished,
        labels_of(labels),
        switches_of(labels),
        mode_runs=runs_of(labels),
        stopped=stopped,
    )


def length_class(length: int, floor: int) -> int:
    level = 0
    while length >= floor << level:
        level += 1
    return level


def pick_group(
    pool: Sequence[_Slot], rows: int, capacity: int, cap: int
) -> tuple[list[_Slot], bool]:
    floor = max(1, capacity // rows // 2)
    classes: dict[int, list[_Slot]] = {}
    for slot in pool:
        classes.setdefault(length_class(slot.length, floor), []).append(slot)
    first: list[_Slot] = []
    for level in sorted(classes, reverse=True):
        members = sorted(classes[level], key=lambda s: (-s.length, s.order))
        head = members[0].length
        need = max(cap - len(s.new) for s in members)
        ahead = min(need, max(MIN_CALL, head // 2))
        fit = max(1, min(rows, capacity // (head + ahead)))
        if not first:
            first = members[:fit]
        if len(members) >= fit:
            return members[:fit], True
    return first, False


@torch.no_grad()
def _generate_refill(
    runner: Runner,
    texts: TokenTexts,
    problems: Sequence[Problem],
    settings: GenSettings,
    step: int,
    on_done: Callable[[Generation], None],
    rows: int | None = None,
) -> int:
    tok = runner.tokenizer
    eos = _eos_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    cap = settings.max_new_tokens
    queue = [
        _Slot(
            p,
            list(
                tok(
                    chat_prompt(tok, p, settings.enable_thinking), add_special_tokens=False
                ).input_ids
            ),
            order=len(problems) - 1 - i,
        )
        for i, p in enumerate(reversed(problems))
    ]
    model = runner.models[0]
    capacity = step * (max((len(s.prompt) for s in queue), default=0) + cap)
    rows = max(step, rows or step)
    pool: list[_Slot] = []
    calls = 0
    while queue or pool:
        active, full = pick_group(pool, rows, capacity, cap)
        while not full and queue:
            pool.append(queue.pop())
            active, full = pick_group(pool, rows, capacity, cap)
        width = max(s.length for s in active)
        ids = torch.full((len(active), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, slot in enumerate(active):
            ids[i, width - slot.length :] = torch.tensor(slot.prompt + slot.new, dtype=torch.long)
            mask[i, width - slot.length :] = 1
        budgets = [cap - len(s.new) for s in active]
        waiting = bool(queue) or len(pool) > len(active)
        low = int(len(active) * REFILL_BELOW) if waiting else -1
        written = [len(s.new) for s in active] if settings.stop_loops else None
        stop = _StopOrRefill(width, budgets, eos, low, runner.device, written)
        room = min(max(budgets), max(1, capacity // len(active) - width))
        config = GenerationConfig(
            max_new_tokens=room,
            do_sample=settings.do_sample,
            temperature=settings.temperature if settings.do_sample else None,
            top_p=settings.top_p if settings.do_sample else None,
            top_k=settings.top_k if settings.do_sample else None,
            eos_token_id=eos,
            pad_token_id=pad,
            prefill_chunk_size=PREFILL_CHUNK,
        )
        torch.manual_seed(settings.seed + calls)
        inputs = {
            "input_ids": ids.to(runner.device),
            "attention_mask": mask.to(runner.device),
            "generation_config": config,
            "stopping_criteria": StoppingCriteriaList([stop]),
            "cache": roomy_cache(width + room),
        }
        if settings.decode == "graph":
            out = _graph_decode(runner, texts, inputs, room, pad)
        elif runner.routing is None:
            out = model.generate(**inputs)
        else:
            with mode_tracking(model, runner.routing, texts):
                out = model.generate(**inputs)
        calls += 1
        for i, slot in enumerate(active):
            limit = stop.loop_at.get(i, budgets[i])
            kept, finished = _trim(out[i, width : width + limit].tolist(), set(eos))
            slot.new += kept
            looped = i in stop.loop_at and not finished
            if finished or looped or len(slot.new) >= cap:
                on_done(_finish(slot, finished, texts, tok, "loop" if looped else None))
                pool.remove(slot)
    return calls


def _graph_decode(
    runner: Runner, texts: TokenTexts, inputs: dict[str, Any], room: int, pad: int
) -> torch.Tensor:
    model = runner.models[0]
    stop = inputs["stopping_criteria"][0]
    args = (model, inputs["input_ids"], inputs["attention_mask"], room, pad, stop)
    if runner.routing is None:
        return greedy_generate(*args)
    with mode_tracking(model, runner.routing, texts) as hook:
        return greedy_generate(*args, routing=runner.routing, hook=hook)


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
        runs_of(labels),
    )


def kv_bytes_per_token(models: Sequence[Qwen3ForCausalLM]) -> int:
    total = 0
    for m in models:
        c = m.config
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        kv_heads = c.num_key_value_heads or c.num_attention_heads
        total += 2 * c.num_hidden_layers * kv_heads * head_dim * m.dtype.itemsize
    return total


def sequence_bytes(models: Sequence[Qwen3ForCausalLM], prompt_tokens: int, new_tokens: int) -> int:
    heads = max(m.config.num_attention_heads for m in models)
    kv = kv_bytes_per_token(models) * (prompt_tokens + new_tokens)
    return kv + heads * prompt_tokens * prompt_tokens * 4


def fit_batch(wanted: int, per_sequence: int, free: int, auto: bool) -> int:
    fits = int(free * KV_FRACTION) // per_sequence
    if fits >= wanted:
        return wanted
    if auto and fits >= 1:
        return fits
    need = wanted * per_sequence / GiB
    budget = free * KV_FRACTION / GiB
    advice = f"pass --batch-size {fits} or auto" if fits else "lower --max-new-tokens"
    raise RuntimeError(
        f"batch {wanted} needs about {need:.1f} GiB of KV cache and prefill attention, but "
        f"{budget:.1f} GiB of the {free / GiB:.1f} GiB free on the GPU is budgeted for it; {advice}"
    )


def resolve_batch(runner: Runner, problems: Sequence[Problem], settings: GenSettings) -> int:
    auto = settings.batch_size == 0
    wanted = 1 if runner.separate else (AUTO_BATCH_CAP if auto else settings.batch_size)
    wanted = max(1, min(wanted, len(problems)))
    if runner.device.type != "cuda" or not problems:
        return wanted
    tok = runner.tokenizer
    longest = max(
        len(tok(chat_prompt(tok, p, settings.enable_thinking), add_special_tokens=False).input_ids)
        for p in problems
    )
    per_sequence = sequence_bytes(runner.models, longest, settings.max_new_tokens)
    free, _ = torch.cuda.mem_get_info(runner.device)
    return fit_batch(wanted, per_sequence, free, auto and not runner.separate)


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
    step = resolve_batch(runner, problems, settings)
    results: list[Generation] = []

    def done(g: Generation) -> None:
        results.append(g)
        if on_done is not None:
            on_done(g)

    if settings.decode == "graph" and (settings.do_sample or runner.separate):
        raise ValueError("--decode graph is greedy and single-model; pass --greedy, not --separate")
    refill = settings.refill and not runner.separate
    refill = refill and (len(problems) > step or settings.stop_loops or settings.decode == "graph")
    rows = min(AUTO_BATCH_CAP, len(problems)) if settings.batch_size == 0 else step
    calls = 0
    fallbacks = len(FALLBACKS)
    start = time.perf_counter()
    if refill:
        calls = _generate_refill(runner, texts, problems, settings, step, done, rows)
    else:
        for i in range(0, len(problems), step):
            chunk = problems[i : i + step]
            if runner.separate:
                batch = [_generate_separate(runner, texts, chunk[0], settings)]
            else:
                batch = _generate_batch(runner, texts, chunk, settings, settings.seed + i)
            calls += 1
            for g in batch:
                done(g)
    elapsed = time.perf_counter() - start
    tokens = sum(g.new_tokens for g in results)
    perf = {
        "seconds": round(elapsed, 2),
        "new_tokens": tokens,
        "tokens_per_s": round(tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "batch_size": step,
        "refill": refill,
        "refill_rows": rows if refill else None,
        "loop_stops": sum(g.stopped == "loop" for g in results),
        "generate_calls": calls,
        "attention_math_fallbacks": len(FALLBACKS) - fallbacks,
        **peak_memory_gib(runner.device),
    }
    return results, perf


def settings_dict(settings: GenSettings) -> dict[str, Any]:
    return asdict(settings)

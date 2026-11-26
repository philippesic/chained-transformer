from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from chained_transformer.checkpoint import convert
from chained_transformer.modes import Mode
from chained_transformer.surgery import Routing, SurgeryConfig
from chained_transformer.train import IGNORE, Batch, make_batch, token_losses
from chained_transformer.upgrade import (
    RetrainConfig,
    Row,
    measure,
    mode_tensors,
    retrain_mode,
    save_mode_pack,
)

THINK, CODE, ANSWER = int(Mode.THINK), int(Mode.CODE), int(Mode.ANSWER)
VOCAB = 64
OPEN, CLOSE, EOS = 61, 62, 63
PROMPT_LEN, THINK_LEN, CODE_LEN, ANSWER_LEN = 4, 16, 12, 4
CODE_STEP = {"v1": 3, "v2": 5}
THINK_AT = slice(PROMPT_LEN, PROMPT_LEN + THINK_LEN)
OPEN_AT = PROMPT_LEN + THINK_LEN
CODE_AT = slice(OPEN_AT + 1, OPEN_AT + 1 + CODE_LEN)
CLOSE_AT = OPEN_AT + 1 + CODE_LEN
ANSWER_AT = slice(CLOSE_AT + 1, CLOSE_AT + 1 + ANSWER_LEN)
EOS_AT = CLOSE_AT + 1 + ANSWER_LEN
LENGTH = EOS_AT + 1


def held_out(p0: int, p1: int) -> bool:
    return (p0 + p1) % 5 == 0


def trace(prompt: list[int], rule: str) -> list[int]:
    think = [10 + (prompt[0] + i) % 20 for i in range(THINK_LEN)]
    code = [30 + (prompt[1] + prompt[2]) % 16]
    while len(code) < CODE_LEN:
        code.append(30 + (code[-1] - 30 + CODE_STEP[rule]) % 16)
    return [*prompt, *think, OPEN, *code, CLOSE, *answer_for(code[-1]), EOS]


def answer_for(last_code: int) -> list[int]:
    return [46 + (last_code - 30 + j) % 15 for j in range(ANSWER_LEN)]


def labels_for(ids: list[int]) -> list[int]:
    labels = [THINK] * (OPEN_AT + 1) + [CODE] * (CODE_LEN + 1) + [ANSWER] * (ANSWER_LEN + 1)
    assert len(labels) == len(ids)
    return labels


def make_row(prompt: list[int], rule: str) -> Row:
    ids = trace(prompt, rule)
    labels = labels_for(ids)
    return {
        "input_ids": ids,
        "mode_ids": [*labels[1:], ANSWER],
        "loss_mask": [0] * PROMPT_LEN + [1] * (len(ids) - PROMPT_LEN),
    }


def make_rows(n: int, rule: str, test: bool, seed: int) -> list[Row]:
    rng = random.Random(seed)
    rows: list[Row] = []
    while len(rows) < n:
        prompt = [rng.randrange(10) for _ in range(PROMPT_LEN)]
        if held_out(prompt[0], prompt[1]) == test:
            rows.append(make_row(prompt, rule))
    return rows


def tiny_base(seed: int) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg).float().eval()


def stack(rows: Sequence[Row], only: int | None) -> Batch:
    parts = [make_batch(r, LENGTH, torch.device("cpu"), only) for r in rows]
    return Batch(
        torch.cat([p.input_ids for p in parts]),
        torch.cat([p.mode_ids for p in parts]),
        torch.cat([p.targets for p in parts]),
        torch.cat([p.target_labels for p in parts]),
    )


def fit(
    model: Qwen3ForCausalLM,
    routing: Routing | None,
    rows: Sequence[Row],
    params: list[nn.Parameter],
    steps: int,
    lr: float,
    batch_rows: int,
    only: int | None,
    seed: int,
) -> list[float]:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        batch = stack(rng.sample(list(rows), batch_rows), only)
        if routing is not None:
            routing.set(batch.mode_ids)
        per_token = token_losses(model, batch, 4096)
        loss = per_token[batch.targets.reshape(-1) != IGNORE].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return losses


@torch.no_grad()
def greedy(model: Qwen3ForCausalLM, routing: Routing | None, prompts: torch.Tensor) -> torch.Tensor:
    ids = prompts
    modes = torch.full_like(ids, THINK)
    for _ in range(LENGTH - PROMPT_LEN):
        if routing is not None:
            routing.set(modes)
        nxt = model(input_ids=ids, use_cache=False).logits[:, -1].argmax(-1)
        mode = torch.where(nxt == OPEN, CODE, torch.where(nxt == CLOSE, ANSWER, modes[:, -1]))
        ids = torch.cat([ids, nxt[:, None]], 1)
        modes = torch.cat([modes, mode[:, None]], 1)
    return ids


def generation_scores(
    model: Qwen3ForCausalLM, routing: Routing | None, rows: Sequence[Row], rule: str
) -> dict[str, float]:
    prompts = torch.tensor([r["input_ids"][:PROMPT_LEN] for r in rows])
    out = greedy(model, routing, prompts).tolist()
    counts = dict.fromkeys(("think", "code", "answer_consistent", "trace"), 0)
    for got in out:
        want = trace(got[:PROMPT_LEN], rule)
        think = got[THINK_AT] == want[THINK_AT] and got[OPEN_AT] == OPEN
        code = got[CODE_AT] == want[CODE_AT] and got[CLOSE_AT] == CLOSE
        answer = (
            got[CLOSE_AT] == CLOSE
            and got[ANSWER_AT] == answer_for(got[CLOSE_AT - 1])
            and got[EOS_AT] == EOS
        )
        counts["think"] += think
        counts["code"] += code
        counts["answer_consistent"] += answer
        counts["trace"] += got == want
    return {k: round(v / len(rows), 4) for k, v in counts.items()}


def evaluate(
    model: Qwen3ForCausalLM,
    routing: Routing | None,
    tests: dict[str, list[Row]],
    reference: tuple[Qwen3ForCausalLM, Routing | None] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rule, rows in tests.items():
        teacher = measure(model, routing, rows, LENGTH, reference)
        entry: dict[str, Any] = {
            "generation": generation_scores(model, routing, rows, rule),
            "teacher_forced": {
                label: {k: parts["all"][k] for k in ("accuracy", "nll", "kl_from_reference")}
                for label, parts in teacher.items()
            },
        }
        if reference is not None:
            entry["max_abs_logprob_diff_before_first_code_position"] = max(
                parts["before_code"]["max_abs_logprob_diff"] for parts in teacher.values()
            )
        result[rule] = entry
    return result


def changed_params(child: nn.Module, parent: nn.Module) -> int:
    before = dict(parent.named_parameters())
    return sum(p.numel() for n, p in child.named_parameters() if not torch.equal(p, before[n]))


def mlp_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if ".mlp." in n]


def run_seed(seed: int, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    train_v1 = make_rows(args.train_rows, "v1", False, 100 + seed)
    train_v2 = make_rows(args.train_rows, "v2", False, 200 + seed)
    tests = {
        "v1": make_rows(args.test_rows, "v1", True, 300 + seed),
        "v2": make_rows(args.test_rows, "v2", True, 400 + seed),
    }
    base = tiny_base(seed)
    modes = convert(copy.deepcopy(base), SurgeryConfig(variant="full"), "tiny-synthetic", seed)
    plain = copy.deepcopy(base)
    pre = {"steps": args.pre_steps, "lr": args.pre_lr, "batch_rows": args.batch_rows}
    timings: dict[str, float] = {}
    started = time.perf_counter()
    fit(modes, modes.routing, train_v1, list(modes.parameters()), seed=seed, only=None, **pre)
    fit(plain, None, train_v1, list(plain.parameters()), seed=seed, only=None, **pre)
    timings["pretrain_s"] = round(time.perf_counter() - started, 1)

    up = {"steps": args.up_steps, "lr": args.up_lr, "batch_rows": args.batch_rows}
    arms: dict[str, tuple[Qwen3ForCausalLM, Routing | None, Qwen3ForCausalLM]] = {}

    def upgrade(name: str, parent: Qwen3ForCausalLM, step: Callable[[Any], object]) -> None:
        child = copy.deepcopy(parent)
        t0 = time.perf_counter()
        step(child)
        timings[f"{name}_s"] = round(time.perf_counter() - t0, 1)
        arms[name] = (child, getattr(child, "routing", None), parent)

    def retrain(loss_on: str) -> Callable[[Any], object]:
        cfg = RetrainConfig(
            mode=CODE,
            steps=args.up_steps,
            lr=args.up_lr,
            batch_rows=args.batch_rows,
            seq_len=LENGTH,
            loss_on="mode" if loss_on == "mode" else "all",
            seed=seed,
        )
        return lambda m: retrain_mode(m, m.routing, train_v2, cfg)

    upgrade("modes_code_mlp_only__code_loss", modes, retrain("mode"))
    upgrade("modes_code_mlp_only__all_loss", modes, retrain("all"))
    upgrade(
        "modes_all_params__code_loss",
        modes,
        lambda m: fit(m, m.routing, train_v2, list(m.parameters()), only=CODE, seed=seed, **up),
    )
    upgrade(
        "plain_all_params__code_loss",
        plain,
        lambda m: fit(m, None, train_v2, list(m.parameters()), only=CODE, seed=seed, **up),
    )
    upgrade(
        "plain_mlp_only__code_loss",
        plain,
        lambda m: fit(m, None, train_v2, mlp_params(m), only=CODE, seed=seed, **up),
    )
    upgrade(
        "plain_all_params__all_loss",
        plain,
        lambda m: fit(m, None, train_v2, list(m.parameters()), only=None, seed=seed, **up),
    )

    upgraded = arms["modes_code_mlp_only__code_loss"][0]
    pack = save_mode_pack(upgraded, CODE, out / f"pack-seed{seed}", {"demo_seed": seed})
    results: dict[str, Any] = {
        "before_upgrade": {
            "modes": evaluate(modes, modes.routing, tests, None),
            "plain": evaluate(plain, None, tests, None),
        },
        "after_upgrade": {
            name: {
                "changed_params": changed_params(child, parent),
                **evaluate(child, routing, tests, (parent, getattr(parent, "routing", None))),
            }
            for name, (child, routing, parent) in arms.items()
        },
        "code_pack": {"params": pack["params"], "tensors": pack["tensors"]},
        "code_mode_params": sum(t.numel() for t in mode_tensors(modes, CODE).values()),
        "timings": timings,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny CPU demo of a single-mode code upgrade")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--train-rows", type=int, default=4000)
    parser.add_argument("--test-rows", type=int, default=256)
    parser.add_argument("--pre-steps", type=int, default=1500)
    parser.add_argument("--pre-lr", type=float, default=3e-3)
    parser.add_argument("--up-steps", type=int, default=300)
    parser.add_argument("--up-lr", type=float, default=1e-3)
    parser.add_argument("--batch-rows", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("artifacts/upgrade-demo"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "task": {
            "length": LENGTH,
            "code_step": CODE_STEP,
            "held_out": "test rows use (prompt[0], prompt[1]) pairs with sum % 5 == 0",
        },
        "args": {k: str(v) for k, v in vars(args).items()},
        "seeds": {str(s): run_seed(s, args, args.out) for s in args.seeds},
    }
    text = json.dumps(report, indent=2) + "\n"
    (args.out / "report.json").write_text(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3ForCausalLM

from chained_transformer.checkpoint import ChainedQwen3ForCausalLM, convert
from chained_transformer.modes import Mode
from chained_transformer.surgery import SurgeryConfig, Variant
from chained_transformer.upgrade import (
    LossOn,
    RetrainConfig,
    load_mode_pack,
    measure,
    mode_tensors,
    retrain_mode,
    save_mode_pack,
    swap_mode,
    trunk_fingerprint,
)

THINK, CODE, ANSWER = int(Mode.THINK), int(Mode.CODE), int(Mode.ANSWER)


def _convert(tiny_model: Qwen3ForCausalLM, variant: Variant) -> ChainedQwen3ForCausalLM:
    model = convert(copy.deepcopy(tiny_model), SurgeryConfig(variant=variant, lora_rank=4), "tiny")
    return model.eval()


def _row(seed: int) -> dict[str, list[int]]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 64, (40,), generator=g).tolist()
    modes = [THINK] * 14 + [CODE] * 14 + [ANSWER] * 12
    mask = [0] * 4 + [1] * 36
    return {"input_ids": ids, "mode_ids": modes, "loss_mask": mask}


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


@pytest.mark.parametrize("variant", ["full", "lora"])
def test_retrain_moves_only_the_chosen_mode(tiny_model: Qwen3ForCausalLM, variant: Variant) -> None:
    model = _convert(tiny_model, variant)
    before = _snapshot(model)
    code = set(mode_tensors(model, CODE))
    rows = [_row(i) for i in range(4)]
    log = retrain_mode(model, model.routing, rows, RetrainConfig(steps=3, lr=1e-2, batch_rows=2))
    assert log.trainable == sum(before[n].numel() for n in code)
    assert len(log.losses) == 3
    after = _snapshot(model)
    moved = {n for n in before if not torch.equal(before[n], after[n])}
    assert moved and moved <= code
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("loss_on", ["mode", "all"])
def test_code_upgrade_leaves_tokens_before_code_untouched(
    tiny_model: Qwen3ForCausalLM, loss_on: LossOn
) -> None:
    reference = _convert(tiny_model, "full")
    model = copy.deepcopy(reference)
    rows = [_row(i) for i in range(4)]
    cfg = RetrainConfig(steps=3, lr=1e-2, batch_rows=2, loss_on=loss_on)
    retrain_mode(model, model.routing, rows, cfg)
    report = measure(model, model.routing, rows, reference=(reference, reference.routing))
    assert report["think"]["before_code"]["max_abs_logprob_diff"] == 0.0
    assert report["think"]["before_code"]["kl_from_reference"] == 0.0
    assert report["code"]["after_code"]["max_abs_logprob_diff"] > 0.0
    assert report["think"]["before_code"]["tokens"] == 4 * 11
    assert report["think"]["after_code"]["tokens"] == 0
    assert report["code"]["after_code"]["tokens"] == 4 * 14
    assert report["answer"]["after_code"]["tokens"] == 4 * 11


def test_measure_against_itself_is_zero(tiny_model: Qwen3ForCausalLM) -> None:
    model = _convert(tiny_model, "lora")
    report = measure(model, model.routing, [_row(0)], reference=(model, model.routing))
    for label in ("think", "code", "answer"):
        assert report[label]["all"]["max_abs_logprob_diff"] == 0.0
        assert 0.0 <= report[label]["all"]["accuracy"] <= 1.0


@pytest.mark.parametrize("variant", ["full", "lora"])
def test_mode_pack_round_trip(
    tmp_path: Path, tiny_model: Qwen3ForCausalLM, variant: Variant
) -> None:
    upgraded = _convert(tiny_model, variant)
    cfg = RetrainConfig(steps=2, lr=1e-2, batch_rows=1)
    retrain_mode(upgraded, upgraded.routing, [_row(1)], cfg)
    header = save_mode_pack(upgraded, CODE, tmp_path / "pack", {"note": "test"})
    assert json.loads((tmp_path / "pack" / "mode.json").read_text()) == header
    assert header["variant"] == variant and header["mode"] == "code" and header["note"] == "test"
    target = _convert(tiny_model, variant)
    assert trunk_fingerprint(target) == header["trunk_sha256"]
    tensors, loaded = load_mode_pack(tmp_path / "pack")
    copied = swap_mode(target, CODE, tensors, loaded)
    assert copied == header["params"]
    ids = torch.randint(0, 64, (1, 20), generator=torch.Generator().manual_seed(3))
    modes = torch.tensor([[THINK] * 6 + [CODE] * 8 + [ANSWER] * 6])
    with torch.no_grad():
        upgraded.routing.set(modes)
        want = upgraded(input_ids=ids).logits
        target.routing.set(modes)
        got = target(input_ids=ids).logits
    assert torch.equal(got, want)


def test_swap_rejects_mismatches(tmp_path: Path, tiny_model: Qwen3ForCausalLM) -> None:
    full = _convert(tiny_model, "full")
    save_mode_pack(full, CODE, tmp_path / "pack")
    tensors, header = load_mode_pack(tmp_path / "pack")
    with pytest.raises(ValueError, match="variant"):
        swap_mode(_convert(tiny_model, "lora"), CODE, tensors, header)
    with pytest.raises(ValueError, match="mode"):
        swap_mode(_convert(tiny_model, "full"), ANSWER, tensors, header)
    other = copy.deepcopy(tiny_model)
    with torch.no_grad():
        other.model.embed_tokens.weight.add_(0.01)
    shifted = _convert(other, "full")
    with pytest.raises(ValueError, match="trunk"):
        swap_mode(shifted, CODE, tensors, header)
    assert swap_mode(shifted, CODE, tensors, header, check_trunk=False) == header["params"]
    with pytest.raises(ValueError, match="do not match"):
        swap_mode(full, CODE, dict(list(tensors.items())[1:]))
    bad = {n: t[..., :-1] if t.dim() > 1 else t for n, t in tensors.items()}
    with pytest.raises(ValueError, match="shape"):
        swap_mode(full, CODE, bad)


def test_moe_has_no_mode_pack(tiny_model: Qwen3ForCausalLM) -> None:
    with pytest.raises(ValueError, match="per-mode"):
        mode_tensors(_convert(tiny_model, "moe"), CODE)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_cli_retrain_swap_measure(tmp_path: Path, tiny_model: Qwen3ForCausalLM, dtype: str) -> None:
    from datasets import Dataset

    from chained_transformer.checkpoint import save_checkpoint
    from chained_transformer.upgrade import main

    rows = [_row(i) for i in range(6)]
    Dataset.from_list(rows).save_to_disk(str(tmp_path / "data"))
    save_checkpoint(_convert(tiny_model, "full"), tmp_path / "ckpt")
    cpu = ["--device", "cpu", "--dtype", dtype]
    data = ["--data", str(tmp_path / "data"), "--seq-len", "64"]
    pack = tmp_path / "pack"
    main(
        ["retrain", "--checkpoint", str(tmp_path / "ckpt"), *cpu, *data, "--steps", "2"]
        + ["--batch-rows", "2", "--lr", "1e-2", "--out", str(pack)]
    )
    header = json.loads((pack / "mode.json").read_text())
    assert header["mode"] == "code" and header["rows"] == 6
    assert len((pack / "train.jsonl").read_text().splitlines()) == 2
    main(
        ["swap", "--checkpoint", str(tmp_path / "ckpt"), *cpu, "--pack", str(pack)]
        + ["--out", str(tmp_path / "swapped")]
    )
    report = tmp_path / "measure.json"
    main(
        ["measure", "--checkpoint", str(tmp_path / "swapped"), *cpu, *data]
        + ["--reference", str(tmp_path / "ckpt"), "--out", str(report)]
    )
    by_label = json.loads(report.read_text())["by_label"]
    assert by_label["think"]["before_code"]["max_abs_logprob_diff"] == 0.0
    assert by_label["code"]["all"]["max_abs_logprob_diff"] > 0.0

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    Qwen3Config,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from chained_transformer.model import Built
from chained_transformer.modes import (
    ASSISTANT_ROLE,
    FENCE_CHAR,
    MIN_FENCE,
    MODE_NAMES,
    THINK_CLOSE,
    TURN_CLOSE,
    TURN_OPEN,
    Mode,
)
from chained_transformer.surgery import SurgeryConfig, Variant, apply_surgery

MODEL_TYPE = "chained_qwen3"
TRACKER_VERSION = 1
VARIANTS: tuple[Variant, ...] = ("full", "lora", "moe")

QWEN3_SHAPES: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-0.6B": {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    },
    "Qwen/Qwen3-1.7B": {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    },
}


def mode_switch_spec(prompt_mode: Mode = Mode.THINK) -> dict[str, Any]:
    return {
        "tracker_version": TRACKER_VERSION,
        "think_close": THINK_CLOSE,
        "turn_open": TURN_OPEN,
        "turn_close": TURN_CLOSE,
        "assistant_role": ASSISTANT_ROLE,
        "fence_char": FENCE_CHAR,
        "min_fence": MIN_FENCE,
        "prompt_mode": MODE_NAMES[prompt_mode],
        "assistant_start_mode": MODE_NAMES[Mode.THINK],
    }


@strict
class ChainedQwen3Config(Qwen3Config):
    model_type = MODEL_TYPE
    mode_surgery: dict | None = None
    mode_names: list[str] | None = None
    mode_switch: dict | None = None
    base_model: str | None = None

    def surgery(self) -> SurgeryConfig:
        return SurgeryConfig(**(self.mode_surgery or {}))


class ChainedQwen3ForCausalLM(Qwen3ForCausalLM):
    config_class = ChainedQwen3Config
    _can_compile_fullgraph = False

    def __init__(self, config: ChainedQwen3Config) -> None:
        super().__init__(config)
        self.routing = apply_surgery(self, config.surgery())


def register() -> None:
    AutoConfig.register(MODEL_TYPE, ChainedQwen3Config, exist_ok=True)
    AutoModelForCausalLM.register(ChainedQwen3Config, ChainedQwen3ForCausalLM, exist_ok=True)


register()


def chained_config(
    base: Qwen3Config,
    surgery: SurgeryConfig,
    base_model: str | None = None,
    prompt_mode: Mode = Mode.THINK,
) -> ChainedQwen3Config:
    fields = {k: v for k, v in base.to_dict().items() if k not in ("model_type", "mode_surgery")}
    fields.pop("architectures", None)
    fields.pop("transformers_version", None)
    if surgery.num_modes != len(MODE_NAMES):
        raise ValueError(f"num_modes={surgery.num_modes} but the tracker has {len(MODE_NAMES)}")
    config = ChainedQwen3Config(
        **fields,
        mode_surgery=surgery.to_dict(),
        mode_names=list(MODE_NAMES),
        mode_switch=mode_switch_spec(prompt_mode),
        base_model=base_model,
    )
    config.architectures = [ChainedQwen3ForCausalLM.__name__]
    return config


def shape_config(name: str) -> Qwen3Config:
    if name not in QWEN3_SHAPES:
        raise KeyError(f"no recorded shape for {name}; known: {', '.join(QWEN3_SHAPES)}")
    return Qwen3Config(**QWEN3_SHAPES[name])


def _set_tensor(module: nn.Module, name: str, value: torch.Tensor) -> None:
    owner_name, _, leaf = name.rpartition(".")
    owner = module.get_submodule(owner_name) if owner_name else module
    owner._buffers[leaf] = value


def convert(
    base: Qwen3ForCausalLM,
    surgery: SurgeryConfig,
    base_model: str | None = None,
    seed: int = 0,
) -> ChainedQwen3ForCausalLM:
    config = chained_config(base.config, surgery, base_model)
    buffers = {n: b for n, b in base.named_buffers()}
    torch.manual_seed(seed)
    apply_surgery(base, surgery)
    state = base.state_dict()
    with torch.device("meta"):
        chained = ChainedQwen3ForCausalLM(config)
    chained.load_state_dict(state, strict=True, assign=True)
    for name, value in buffers.items():
        if name not in state:
            _set_tensor(chained, name, value)
    chained.tie_weights()
    if (
        config.tie_word_embeddings
        and chained.lm_head.weight is not chained.model.embed_tokens.weight
    ):
        raise RuntimeError("lm_head is not tied to embed_tokens after conversion")
    leftover = [n for n, t in [*chained.named_parameters(), *chained.named_buffers()] if t.is_meta]
    if leftover:
        raise RuntimeError(f"tensors left on the meta device after conversion: {leftover[:5]}")
    chained.generation_config = base.generation_config
    chained.train(base.training)
    return chained


def convert_from_hub(
    name: str,
    variant: Variant,
    dtype: torch.dtype = torch.float32,
    surgery: SurgeryConfig | None = None,
    seed: int = 0,
) -> ChainedQwen3ForCausalLM:
    from chained_transformer.model import load_base

    base = load_base(name, dtype=dtype, device="cpu")
    return convert(base, replace(surgery or SurgeryConfig(), variant=variant), name, seed)


def save_checkpoint(
    model: Qwen3ForCausalLM,
    path: Path,
    tokenizer: PreTrainedTokenizerBase | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(path))
    if extra is not None:
        (path / "training_state.json").write_text(json.dumps(extra, indent=2) + "\n")


def load_checkpoint(
    path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    attn_implementation: str = "sdpa",
) -> Built:
    model = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=dtype, attn_implementation=attn_implementation
    )
    nn.Module.to(model, torch.device(device))
    if isinstance(model, ChainedQwen3ForCausalLM):
        return Built(model, model.routing)
    if isinstance(model, Qwen3ForCausalLM):
        return Built(model, None)
    raise TypeError(f"{path} loaded as {type(model).__name__}, expected a Qwen3 model")


def is_chained(path: str | Path) -> bool:
    config = Path(path) / "config.json"
    return config.exists() and json.loads(config.read_text()).get("model_type") == MODEL_TYPE


def mode_slice(model: ChainedQwen3ForCausalLM, mode: int) -> dict[str, torch.Tensor]:
    from chained_transformer.surgery import mode_parameters

    wanted = {id(p) for p in mode_parameters(model, mode)}
    return {n: p.detach() for n, p in model.named_parameters() if id(p) in wanted}


def meta_counts(config: ChainedQwen3Config | Qwen3Config) -> dict[str, int]:
    with torch.device("meta"):
        model: nn.Module = (
            ChainedQwen3ForCausalLM(config)
            if isinstance(config, ChainedQwen3Config)
            else Qwen3ForCausalLM(config)
        )
    total = sum(p.numel() for p in model.parameters())
    mlp = sum(p.numel() for n, p in model.named_parameters() if ".mlp." in n)
    per_mode = 0
    if isinstance(model, ChainedQwen3ForCausalLM):
        from chained_transformer.surgery import mode_parameters

        per_mode = sum(p.numel() for p in mode_parameters(model, 0))
    return {"total": total, "mlp": mlp, "trainable_per_mode": per_mode}


def plan(name: str, out: Path | None) -> dict[str, Any]:
    base = shape_config(name)
    report: dict[str, Any] = {"base_model": name, "base": meta_counts(base), "variants": {}}
    for variant in VARIANTS:
        config = chained_config(base, SurgeryConfig(variant=variant), name)
        counts = meta_counts(config)
        report["variants"][variant] = {
            **counts,
            "bf16_weights_gib": round(counts["total"] * 2 / 2**30, 3),
        }
        if out is not None:
            target = out / variant
            target.mkdir(parents=True, exist_ok=True)
            config.save_pretrained(str(target))
    if out is not None:
        (out / "params.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert, inspect and plan mode checkpoints")
    sub = parser.add_subparsers(dest="command", required=True)
    conv = sub.add_parser("convert", help="convert a Hub Qwen3 model into one variant")
    conv.add_argument("--base", default="Qwen/Qwen3-0.6B")
    conv.add_argument("--variant", choices=VARIANTS, required=True)
    conv.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    conv.add_argument("--lora-rank", type=int, default=64)
    conv.add_argument("--moe-top-k", type=int, default=2)
    conv.add_argument("--seed", type=int, default=0)
    conv.add_argument("--out", type=Path, required=True)
    shapes = sub.add_parser("plan", help="write configs and parameter counts without weights")
    shapes.add_argument("--base", choices=list(QWEN3_SHAPES), required=True)
    shapes.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(plan(args.base, args.out), indent=2))
        return
    from chained_transformer.model import load_tokenizer

    surgery = SurgeryConfig(
        variant=args.variant,
        lora_rank=args.lora_rank,
        lora_alpha=2.0 * args.lora_rank,
        moe_top_k=args.moe_top_k,
    )
    model = convert_from_hub(args.base, args.variant, DTYPES[args.dtype], surgery, args.seed)
    save_checkpoint(model, args.out, load_tokenizer(args.base))
    print(json.dumps({"out": str(args.out), "params": meta_counts(model.config)}, indent=2))


if __name__ == "__main__":
    main()

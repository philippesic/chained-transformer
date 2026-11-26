from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

ArmName = Literal["sft", "full", "lora", "moe", "separate"]
ARM_NAMES: tuple[ArmName, ...] = ("sft", "full", "lora", "moe", "separate")
Optim = Literal["torchao-eager", "torchao-compiled", "chunked"]
Attention = Literal["math-fp32", "fused"]
GiB = float(2**30)
BLOCK = 256
MIN_QUANT_NUMEL = 4096


@dataclass(frozen=True)
class ModelShape:
    vocab: int = 151_936
    hidden: int = 1024
    intermediate: int = 3072
    layers: int = 28
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    tied: bool = True

    @property
    def q_width(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim


QWEN3_0_6B = ModelShape()
QWEN3_1_7B = ModelShape(hidden=2048, intermediate=6144, layers=28, heads=16, kv_heads=8)


@dataclass(frozen=True)
class Setup:
    arm: ArmName
    seq_len: int
    batch: int = 1
    modes: int = 3
    lora_rank: int = 64
    moe_top_k: int = 2
    optimizer: Optim = "chunked"
    attention: Attention = "fused"
    loss_chunk: int = 256
    grad_ckpt: bool = True
    freeze_embeddings: bool = False
    optim_chunk: int = 1 << 22


@dataclass(frozen=True)
class Tensor:
    name: str
    numel: int
    trainable: bool


@dataclass(frozen=True)
class Coefficients:
    norm: float = 8.0
    q_path: float = 12.0
    k_path: float = 12.0
    v_path: float = 2.0
    attn_fused: float = 10.0
    o_in: float = 2.0
    mlp: float = 8.0
    routed_gather: float = 6.0
    moe_gather: float = 6.0
    lora: float = 6.0
    layer_backward: float = 1.6
    math_attn: float = 12.0
    loss_bytes_per_logit: float = 14.0
    torchao_eager_temps: float = 7.0
    chunked_temps: float = 40.0


@dataclass
class Estimate:
    setup: Setup
    shape: ModelShape
    params_total: int
    params_trainable: int
    bytes: dict[str, float] = field(default_factory=dict)
    phases: dict[str, float] = field(default_factory=dict)

    @property
    def peak(self) -> float:
        return max(self.phases.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "setup": asdict(self.setup),
            "params_total": self.params_total,
            "params_trainable": self.params_trainable,
            "gib": {k: round(v / GiB, 3) for k, v in self.bytes.items()},
            "phases_gib": {k: round(v / GiB, 3) for k, v in self.phases.items()},
            "peak_gib": round(self.peak / GiB, 3),
        }


def parameter_tensors(shape: ModelShape, setup: Setup) -> list[Tensor]:
    h, i, q, kv = shape.hidden, shape.intermediate, shape.q_width, shape.kv_width
    trunk = True  # attention, norms and the shared trunk train in every arm
    tensors = [Tensor("embed", shape.vocab * h, not setup.freeze_embeddings)]
    if not shape.tied:
        tensors.append(Tensor("lm_head", shape.vocab * h, not setup.freeze_embeddings))
    tensors.append(Tensor("final_norm", h, trunk))
    copies = setup.modes if setup.arm in ("full", "moe") else 1
    for layer in range(shape.layers):
        p = f"layers.{layer}."
        tensors += [
            Tensor(p + "q_proj", h * q, trunk),
            Tensor(p + "k_proj", h * kv, trunk),
            Tensor(p + "v_proj", h * kv, trunk),
            Tensor(p + "o_proj", q * h, trunk),
            Tensor(p + "q_norm", shape.head_dim, trunk),
            Tensor(p + "k_norm", shape.head_dim, trunk),
            Tensor(p + "input_norm", h, trunk),
            Tensor(p + "post_norm", h, trunk),
        ]
        base_trainable = setup.arm != "lora"
        for c in range(copies):
            for proj in ("gate", "up", "down"):
                tensors.append(Tensor(f"{p}mlp.{c}.{proj}", h * i, base_trainable))
        if setup.arm == "moe":
            tensors.append(Tensor(p + "router", h * setup.modes, True))
        if setup.arm == "lora":
            r = setup.lora_rank
            for m in range(setup.modes):
                for proj, (fan_in, fan_out) in (("gate", (h, i)), ("up", (h, i)), ("down", (i, h))):
                    tensors.append(Tensor(f"{p}lora.{m}.{proj}.a", r * fan_in, True))
                    tensors.append(Tensor(f"{p}lora.{m}.{proj}.b", fan_out * r, True))
    return tensors


def optimizer_state_bytes(t: Tensor) -> float:
    if not t.trainable:
        return 0.0
    if t.numel >= MIN_QUANT_NUMEL and t.numel % BLOCK == 0:
        return 2.0 * t.numel + 8.0 * t.numel / BLOCK
    return 8.0 * t.numel


def saved_layer_bytes(shape: ModelShape, setup: Setup, k: Coefficients) -> float:
    n = setup.batch * setup.seq_len
    h, i, q, kv = shape.hidden, shape.intermediate, shape.q_width, shape.kv_width
    per_token = (
        2 * k.norm * h
        + k.q_path * q
        + k.k_path * kv
        + k.v_path * kv
        + k.attn_fused * q
        + k.o_in * q
    )
    if setup.arm == "moe":
        per_token += k.mlp * i * setup.moe_top_k + k.moe_gather * h * setup.moe_top_k
    else:
        per_token += k.mlp * i
    if setup.arm in ("full", "lora"):
        per_token += k.routed_gather * h
    if setup.arm == "lora":
        per_token += k.lora * 3 * setup.lora_rank
    return per_token * n


def math_attention_bytes(shape: ModelShape, setup: Setup, k: Coefficients) -> float:
    if setup.attention != "math-fp32":
        return 0.0
    return k.math_attn * setup.batch * shape.heads * setup.seq_len**2


def loss_bytes(shape: ModelShape, setup: Setup, k: Coefficients) -> float:
    chunk = min(setup.loss_chunk, setup.batch * setup.seq_len)
    weight_grad = 2.0 * shape.vocab * shape.hidden
    return k.loss_bytes_per_logit * chunk * shape.vocab + weight_grad


def optimizer_temp_bytes(tensors: list[Tensor], setup: Setup, k: Coefficients) -> float:
    largest = max((t.numel for t in tensors if t.trainable), default=0)
    if setup.optimizer == "torchao-eager":
        return k.torchao_eager_temps * 4.0 * largest
    if setup.optimizer == "torchao-compiled":
        return 4.0 * largest
    return k.chunked_temps * min(largest, setup.optim_chunk)


def estimate(shape: ModelShape, setup: Setup, k: Coefficients | None = None) -> Estimate:
    k = k or Coefficients()
    tensors = parameter_tensors(shape, setup)
    total = sum(t.numel for t in tensors)
    trainable = sum(t.numel for t in tensors if t.trainable)
    n = setup.batch * setup.seq_len
    weights = 2.0 * total
    grads = 2.0 * trainable
    states = sum(optimizer_state_bytes(t) for t in tensors)
    layer = saved_layer_bytes(shape, setup, k)
    if setup.grad_ckpt:
        saved = shape.layers * 2.0 * n * shape.hidden + 8.0 * n * shape.hidden
        layer_peak = k.layer_backward * layer + math_attention_bytes(shape, setup, k)
    else:
        saved = shape.layers * (layer + math_attention_bytes(shape, setup, k) / 3.0)
        layer_peak = (k.layer_backward - 1.0) * layer + math_attention_bytes(shape, setup, k)
    loss = loss_bytes(shape, setup, k)
    opt_temp = optimizer_temp_bytes(tensors, setup, k)
    est = Estimate(setup, shape, total, trainable)
    est.bytes = {
        "weights": weights,
        "grads": grads,
        "optimizer_state": states,
        "saved_activations": saved,
        "layer_backward_peak": layer_peak,
        "loss_chunk_peak": loss,
        "optimizer_step_temps": opt_temp,
    }
    resident = weights + states
    est.phases = {
        "forward": resident + saved + layer_peak / k.layer_backward,
        "loss_backward": resident + 2.0 * shape.vocab * shape.hidden + saved + loss,
        "layer_backward": resident + grads + saved + layer_peak,
        "optimizer_step": resident + grads + opt_temp,
    }
    return est


def current_harness(arm: ArmName, seq_len: int, turing: bool) -> Setup:
    return Setup(
        arm=arm,
        seq_len=seq_len,
        optimizer="torchao-eager" if turing else "torchao-compiled",
        attention="math-fp32" if turing else "fused",
        loss_chunk=1024,
    )


def fixed_harness(arm: ArmName, seq_len: int) -> Setup:
    return Setup(arm=arm, seq_len=seq_len)


def table(
    shape: ModelShape, setups: list[Setup], k: Coefficients | None = None
) -> list[dict[str, object]]:
    return [estimate(shape, s, k).to_dict() for s in setups]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Estimate training memory per arm")
    parser.add_argument("--seq-len", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=list(ARM_NAMES))
    parser.add_argument(
        "--harness", choices=["current-turing", "current-ampere", "fixed"], default="fixed"
    )
    parser.add_argument("--loss-chunk", type=int, default=None)
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows = []
    for seq in args.seq_len:
        for arm in args.arms:
            if args.harness == "fixed":
                s = fixed_harness(arm, seq)
            else:
                s = current_harness(arm, seq, args.harness == "current-turing")
            if args.loss_chunk:
                s = replace(s, loss_chunk=args.loss_chunk)
            if args.freeze_embeddings:
                s = replace(s, freeze_embeddings=True)
            rows.append(estimate(QWEN3_0_6B, s).to_dict())
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(
        f"{'arm':<9}{'seq':>7}{'weights':>9}{'grads':>7}{'optim':>7}{'acts':>7}{'peak':>8}  phase"
    )
    for r in rows:
        g = r["gib"]
        ph = r["phases_gib"]
        assert isinstance(g, dict) and isinstance(ph, dict)
        worst = max(ph, key=lambda name: ph[name])
        setup = r["setup"]
        assert isinstance(setup, dict)
        print(
            f"{setup['arm']:<9}{setup['seq_len']:>7}{g['weights']:>9.2f}{g['grads']:>7.2f}"
            f"{g['optimizer_state']:>7.2f}{g['saved_activations']:>7.2f}"
            f"{r['peak_gib']:>8.2f}  {worst}"
        )


if __name__ == "__main__":
    main()

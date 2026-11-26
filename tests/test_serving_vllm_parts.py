from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from transformers import Qwen3ForCausalLM

from chained_transformer.checkpoint import load_checkpoint
from chained_transformer.modes import Mode
from chained_transformer.surgery import ModeMLP, Routing
from serving.export import TOKEN_TEXTS_FILE, export_for_vllm
from serving.tiny import (
    TINY_PROMPTS,
    tiny_chained_model,
    tiny_greedy,
    tiny_pilot_checkpoint,
    tiny_token_texts,
)

vllm = pytest.importorskip("vllm")

NEW_TOKENS = 24
LONG_BASE = [0, 1, *range(6, 30), 4, 2, 4, *range(30, 60), 4, 3, *range(10, 20)]
LONG_PROMPTS = [[*LONG_BASE, k] for k in (20, 21, 22, 23)]
MAX_SEQS = 8
BUDGET = 32
NUM_BLOCKS = 256


@dataclasses.dataclass
class Req:
    req_id: str
    slot: int
    tokens: list[int]
    prompt_len: int
    computed: int
    blocks: list[int]
    max_new: int


class MiniEngine:
    def __init__(self, vllm_config: Any, model: Any) -> None:
        from vllm.model_executor.layers.attention import Attention
        from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadataBuilder
        from vllm.v1.worker.gpu.input_batch import InputBuffers

        from serving.vllm_plugin.state import ModeModelState

        self.config = vllm_config
        self.model = model
        self.block_size = vllm_config.cache_config.block_size
        cpu = torch.device("cpu")
        from vllm.config import get_layers_from_vllm_config

        layers = get_layers_from_vllm_config(vllm_config, Attention)
        self.layer_names = list(layers)
        spec = next(iter(layers.values())).get_kv_cache_spec(vllm_config)
        for layer in layers.values():
            layer.kv_cache = torch.zeros(
                NUM_BLOCKS, spec.num_kv_heads, self.block_size, 2 * spec.head_size, dtype=spec.dtype
            )
        self.builder = CPUAttentionMetadataBuilder(spec, self.layer_names, vllm_config, cpu)
        self.state = ModeModelState(vllm_config, model, None, cpu)
        sched = vllm_config.scheduler_config
        self.buffers = InputBuffers(sched.max_num_seqs, sched.max_num_batched_tokens, cpu)
        self.free_blocks = list(range(NUM_BLOCKS - 1, -1, -1))
        self.free_slots = list(range(sched.max_num_seqs - 1, -1, -1))
        self.prefix_blocks: dict[tuple[int, ...], int] = {}
        self.running: list[Req] = []
        self.done: dict[str, list[int]] = {}
        self.counter = 0

    def cached_prefix(self, tokens: list[int]) -> list[int]:
        blocks = []
        full = (len(tokens) - 1) // self.block_size
        for k in range(full):
            block = self.prefix_blocks.get(tuple(tokens[: (k + 1) * self.block_size]))
            if block is None:
                break
            blocks.append(block)
        return blocks

    def admit(self, tokens: list[int], max_new: int, req_id: str | None = None) -> str:
        from vllm.v1.core.sched.output import NewRequestData

        if req_id is None:
            req_id = f"r{self.counter}"
            self.counter += 1
        blocks = self.cached_prefix(tokens)
        req = Req(
            req_id=req_id,
            slot=self.free_slots.pop(),
            tokens=list(tokens),
            prompt_len=len(tokens),
            computed=len(blocks) * self.block_size,
            blocks=blocks,
            max_new=max_new,
        )
        data = NewRequestData(
            req_id=req_id,
            prompt_token_ids=list(tokens),
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=(list(blocks),),
            num_computed_tokens=req.computed,
            lora_request=None,
            prefill_token_ids=list(tokens),
        )
        self.state.add_request(req.slot, data)
        self.running.append(req)
        return req_id

    def preempt(self, req_id: str) -> tuple[list[int], int]:
        req = next(r for r in self.running if r.req_id == req_id)
        self.running.remove(req)
        self.state.remove_request(req_id)
        self.free_slots.append(req.slot)
        generated = len(req.tokens) - req.prompt_len
        return req.tokens, req.max_new - generated

    def ensure_blocks(self, req: Req, length: int) -> None:
        while len(req.blocks) * self.block_size < length:
            req.blocks.append(self.free_blocks.pop())

    def remember_blocks(self, req: Req) -> None:
        for k in range(req.computed // self.block_size):
            key = tuple(req.tokens[: (k + 1) * self.block_size])
            self.prefix_blocks.setdefault(key, req.blocks[k])

    def step(self) -> None:
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.worker.gpu.input_batch import InputBatch

        budget = BUDGET
        rows: list[tuple[Req, int]] = []
        for req in self.running:
            if budget == 0:
                break
            n = min(len(req.tokens) - req.computed, budget)
            rows.append((req, n))
            budget -= n
        starts = [req.computed for req, _ in rows]
        counts = [n for _, n in rows]
        qsl = np.zeros(len(rows) + 1, dtype=np.int32)
        np.cumsum(counts, out=qsl[1:])
        num_tokens = int(qsl[-1])
        ids, positions, slots = [], [], []
        for (req, n), start in zip(rows, starts, strict=True):
            self.ensure_blocks(req, start + n)
            for pos in range(start, start + n):
                ids.append(req.tokens[pos])
                positions.append(pos)
                slots.append(
                    req.blocks[pos // self.block_size] * self.block_size + pos % self.block_size
                )
        width = max(len(req.blocks) for req, _ in rows)
        table = torch.zeros(len(rows), width, dtype=torch.int32)
        for row, (req, _) in enumerate(rows):
            table[row, : len(req.blocks)] = torch.tensor(req.blocks, dtype=torch.int32)
        seq_lens = torch.tensor(
            [s + n for s, n in zip(starts, counts, strict=True)], dtype=torch.int32
        )
        qsl_t = torch.from_numpy(qsl.copy())
        pos_t = torch.tensor(positions, dtype=torch.int64)
        slot_t = torch.tensor(slots, dtype=torch.int64)
        dummy = InputBatch.make_dummy(len(rows), num_tokens, self.buffers)
        batch = dataclasses.replace(
            dummy,
            req_ids=[req.req_id for req, _ in rows],
            idx_mapping_np=np.array([req.slot for req, _ in rows], dtype=np.intp),
            idx_mapping=torch.tensor([req.slot for req, _ in rows], dtype=torch.int64),
            num_scheduled_tokens=np.array(counts, dtype=np.int32),
            query_start_loc_np=qsl,
            query_start_loc=qsl_t,
            num_computed_tokens_np=np.array(starts, dtype=np.int32),
            seq_lens=seq_lens,
            input_ids=torch.tensor(ids, dtype=torch.int32),
            positions=pos_t,
            logits_indices=qsl_t[1:].long() - 1,
        )
        inputs = self.state.prepare_inputs(batch, None)
        common = CommonAttentionMetadata(
            query_start_loc=qsl_t,
            query_start_loc_cpu=qsl_t,
            seq_lens=seq_lens,
            num_reqs=len(rows),
            num_actual_tokens=num_tokens,
            max_query_len=max(counts),
            max_seq_len=int(seq_lens.max()),
            block_table_tensor=table,
            slot_mapping=slot_t,
            positions=pos_t,
            seq_lens_cpu_upper_bound=seq_lens,
        )
        metadata = self.builder.build(0, common)
        with set_forward_context(
            {name: metadata for name in self.layer_names},
            self.config,
            num_tokens=num_tokens,
            slot_mapping={name: slot_t for name in self.layer_names},
        ):
            hidden = self.model(batch.input_ids, batch.positions, mode_ids=inputs["mode_ids"])
            logits = self.model.compute_logits(hidden[batch.logits_indices])
        picks = logits.argmax(dim=-1).tolist()
        for (req, n), pick in zip(rows, picks, strict=True):
            req.computed += n
            self.remember_blocks(req)
            if req.computed == len(req.tokens):
                req.tokens.append(int(pick))
                if len(req.tokens) - req.prompt_len == req.max_new:
                    self.finish(req)

    def finish(self, req: Req) -> None:
        self.running.remove(req)
        self.state.remove_request(req.req_id)
        self.free_slots.append(req.slot)
        self.done[req.req_id] = req.tokens[req.prompt_len :]

    def run(self, prompts: list[list[int]], max_new: int = NEW_TOKENS) -> list[str]:
        req_ids = [self.admit(p, max_new) for p in prompts]
        while self.running:
            self.step()
        return req_ids


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def tiny() -> tuple[Qwen3ForCausalLM, Routing]:
    return tiny_chained_model()


@pytest.fixture(scope="module")
def exported(
    tiny: tuple[Qwen3ForCausalLM, Routing], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    return export_for_vllm(
        tiny[0], tmp_path_factory.mktemp("tiny-chained"), token_texts=tiny_token_texts()
    )


def make_config(exported: Path, enforce_eager: bool, impl: str) -> Any:
    import vllm.triton_utils as triton_utils
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(
        model=str(exported),
        skip_tokenizer_init=True,
        dtype="float32",
        enforce_eager=enforce_eager,
        max_model_len=256,
        max_num_seqs=MAX_SEQS,
        max_num_batched_tokens=BUDGET,
        block_size=32,
        seed=0,
        hf_overrides={
            "mode_tracking": {
                "prompt_mode": 0,
                "token_texts_file": TOKEN_TEXTS_FILE,
                "mlp_impl": impl,
                "record": True,
            }
        },
    )
    had_triton = triton_utils.HAS_TRITON
    triton_utils.HAS_TRITON = True
    try:
        return args.create_engine_config()
    finally:
        triton_utils.HAS_TRITON = had_triton


def load_model(config: Any, exported: Path) -> tuple[Any, dict[str, torch.Tensor]]:
    from safetensors.torch import load_file
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.utils.torch_utils import set_default_torch_dtype

    from serving.vllm_plugin.model import ChainedQwen3ForCausalLM

    with set_default_torch_dtype(torch.float32):
        model = ChainedQwen3ForCausalLM(vllm_config=config)
    loaded = model.load_weights(iter(load_file(str(exported / "model.safetensors")).items()))
    assert loaded == {name for name, _ in model.named_parameters()}
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    for layer in model.model.layers:
        stacked = layer.mlp.stacked()
        for m, mode in enumerate(layer.mlp.modes):
            assert mode.gate_up_proj.weight.data_ptr() == stacked.gate_up[m].data_ptr()
            assert mode.down_proj.weight.data_ptr() == stacked.down[m].data_ptr()
    process_weights_after_loading(model, config.model_config, torch.device("cpu"))
    return model.eval(), weights


@pytest.fixture(scope="module")
def vllm_model(exported: Path) -> Iterator[tuple[Any, Any, dict[str, torch.Tensor]]]:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.platforms import current_platform

    from serving.vllm_plugin import register

    if not current_platform.is_cpu():
        pytest.skip("the hand-driven engine uses the CPU attention backend")
    register()
    config = make_config(exported, enforce_eager=True, impl="select")
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{free_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1)
        model, weights = load_model(config, exported)
        with torch.no_grad():
            yield config, model, weights
        destroy_model_parallel()
        destroy_distributed_environment()


def set_impl(model: Any, impl: str) -> None:
    from serving.vllm_plugin.model import VllmModeMLP

    for module in model.modules():
        if isinstance(module, VllmModeMLP):
            module.impl = impl


def test_registered_architecture_resolves_to_plugin(exported: Path) -> None:
    from vllm import ModelRegistry
    from vllm.engine.arg_utils import EngineArgs

    from serving.vllm_plugin import ARCHITECTURE, register
    from serving.vllm_plugin.model import ChainedQwen3ForCausalLM

    register()
    register()
    assert ARCHITECTURE in ModelRegistry.get_supported_archs()
    model_config = EngineArgs(
        model=str(exported),
        skip_tokenizer_init=True,
        dtype="float32",
        hf_overrides={"mode_tracking": {"mlp_impl": "loop"}},
    ).create_model_config()
    assert model_config.architectures == [ARCHITECTURE]
    assert model_config.hf_config.mode_tracking["mlp_impl"] == "loop"
    cls, arch = ModelRegistry.resolve_model_cls(model_config.architectures, model_config)
    assert cls is ChainedQwen3ForCausalLM
    assert arch == ARCHITECTURE
    assert cls.get_model_state_cls().__name__ == "ModeModelState"


def test_entry_point_shim_registers_from_its_own_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    from vllm import ModelRegistry

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "serving" / "vllm_entry"))
    monkeypatch.setenv("CHAINED_TRANSFORMER_ROOT", str(root))
    shim = importlib.import_module("chained_vllm_entry")
    shim.register()
    assert str(root) in sys.path
    assert "ChainedQwen3ForCausalLM" in ModelRegistry.get_supported_archs()


def test_loaded_weights_match_checkpoint(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]], tiny: tuple[Qwen3ForCausalLM, Routing]
) -> None:
    weights = vllm_model[2]
    hf = tiny[0]
    for i, layer in enumerate(hf.model.layers):
        mlp = layer.mlp
        assert isinstance(mlp, ModeMLP)
        for m, mode in enumerate(mlp.modes):
            prefix = f"model.layers.{i}.mlp.modes.{m}"
            gate_up = torch.cat(
                [mode.get_parameter("gate_proj.weight"), mode.get_parameter("up_proj.weight")]
            )
            torch.testing.assert_close(
                weights[f"{prefix}.gate_up_proj.weight"], gate_up, rtol=0, atol=0
            )
            torch.testing.assert_close(
                weights[f"{prefix}.down_proj.weight"],
                mode.get_parameter("down_proj.weight"),
                rtol=0,
                atol=0,
            )
    torch.testing.assert_close(
        weights["model.embed_tokens.weight"], hf.model.embed_tokens.weight, rtol=0, atol=0
    )
    model = vllm_model[1]
    for i, layer in enumerate(hf.model.layers):
        mlp = layer.mlp
        assert isinstance(mlp, ModeMLP)
        stacked = model.model.layers[i].mlp.stacked()
        for m, mode in enumerate(mlp.modes):
            gate_up = torch.cat(
                [mode.get_parameter("gate_proj.weight"), mode.get_parameter("up_proj.weight")]
            )
            torch.testing.assert_close(stacked.gate_up[m], gate_up, rtol=0, atol=0)
            torch.testing.assert_close(
                stacked.down[m], mode.get_parameter("down_proj.weight"), rtol=0, atol=0
            )


def triton_runs_here() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"


IMPLS = [
    "select",
    "loop",
    pytest.param(
        "triton",
        marks=pytest.mark.skipif(not triton_runs_here(), reason="needs Triton or its interpreter"),
    ),
]


@pytest.mark.parametrize("impl", IMPLS)
def test_mode_mlp_routes_each_token(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]],
    tiny: tuple[Qwen3ForCausalLM, Routing],
    impl: str,
) -> None:
    from vllm.config import set_current_vllm_config

    config, model, _ = vllm_model
    hf_mlp = tiny[0].model.layers[1].mlp
    assert isinstance(hf_mlp, ModeMLP)
    x = torch.randn(23, 128, generator=torch.Generator().manual_seed(3))
    ids = torch.tensor([i % 3 for i in range(23)][::-1], dtype=torch.int64)
    want = torch.stack([hf_mlp.modes[int(m)](x[i]) for i, m in enumerate(ids)])
    mlp = model.model.layers[1].mlp
    mlp.impl = impl
    with set_current_vllm_config(config):
        got = mlp(x, torch.cat([ids, torch.zeros(9, dtype=torch.int64)]))
        first = mlp(x, None)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=2e-4)
    torch.testing.assert_close(first, hf_mlp.modes[0](x), rtol=1e-4, atol=2e-4)


def test_dummy_batches_get_think_modes(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]],
) -> None:
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    from serving.vllm_plugin.state import ModeModelState

    config, model, _ = vllm_model
    state = ModeModelState(config, model, None, torch.device("cpu"))
    state.mode_ids.fill_(2)
    assert not state.prepare_dummy_inputs(3, 17)["mode_ids"].any()
    buffers = InputBuffers(MAX_SEQS, BUDGET, torch.device("cpu"))
    batch = InputBatch.make_dummy(4, 13, buffers)
    batch.input_ids.copy_(torch.tensor([2, 3, 2, 3, 0, 1, 2, 3, 3, 3, 2, 2, 3], dtype=torch.int32))
    state.mode_ids.fill_(2)
    out = state.prepare_inputs(batch, None)["mode_ids"]
    assert out.shape == (13,)
    assert not out.any()
    assert state.steps == 0


def test_model_refuses_to_run_without_mode_ids(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]],
) -> None:
    from vllm.config import set_current_vllm_config

    config, model, _ = vllm_model
    with set_current_vllm_config(config), pytest.raises(RuntimeError, match="got no mode_ids"):
        model(torch.zeros(3, dtype=torch.long), torch.arange(3))


def test_mode_tracking_rejects_out_of_range_modes(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    from serving.vllm_plugin.state import ModeModelState

    config, model, _ = vllm_model
    state = ModeModelState(config, model, None, torch.device("cpu"))
    batch = InputBatch.make_dummy(2, 5, InputBuffers(MAX_SEQS, BUDGET, torch.device("cpu")))
    monkeypatch.setattr(state.slots, "batch_modes", lambda *a: np.array([0, 1, 2, 3, 0]))
    with pytest.raises(RuntimeError, match="outside 0..2"):
        state.prepare_inputs(batch, None)


def test_runner_report_checks_stacked_weights(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]],
) -> None:
    from types import SimpleNamespace

    from vllm.config import CompilationMode

    from serving.vllm_plugin.model import VllmModeMLP
    from serving.vllm_plugin.state import (
        ModeModelState,
        collect_traces,
        external_id,
        runner_report,
    )

    config, model, _ = vllm_model
    state = ModeModelState(config, model, None, torch.device("cpu"))
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(vllm_config=config, model=model, model_state=state)
    )
    report = runner_report(worker)
    assert report["model_state"] == "ModeModelState"
    assert report["mode_mlps"] == 2
    assert report["stacking_problems"] == []
    assert report["compilation_mode"] in {m.name for m in CompilationMode}
    assert report["cudagraph_mode"] == "NONE"
    assert collect_traces(worker)["traces"] == {}
    assert external_id("3-0a1b2c3d") == "3"
    assert external_id("my-request") == "my-request"
    mlp = next(m for m in model.modules() if isinstance(m, VllmModeMLP))
    weight = mlp.modes[1].get_parameter("down_proj.weight")
    saved = weight.data
    weight.data = torch.ones_like(mlp.stacked().down[1])
    try:
        assert "no longer shares memory" in runner_report(worker)["stacking_problems"][0]
    finally:
        weight.data = saved


def test_mode_state_lists_untested_vllm_features() -> None:
    from types import SimpleNamespace

    from serving.vllm_plugin.state import unsupported_features

    def config(spec: Any = None, ubatch: bool = False, pp: int = 1) -> Any:
        parallel = SimpleNamespace(use_ubatching=ubatch, pipeline_parallel_size=pp)
        return SimpleNamespace(speculative_config=spec, parallel_config=parallel)

    assert unsupported_features(config()) == []
    found = unsupported_features(config(spec=object(), ubatch=True, pp=2))
    assert [f.split(" ")[0] for f in found] == ["speculative", "micro-batching", "pipeline"]
    assert "mode_ids" in found[1]


def test_mode_state_refuses_speculative_decoding(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]],
) -> None:
    import copy
    from types import SimpleNamespace

    from serving.vllm_plugin.state import ModeModelState

    config, model, _ = vllm_model
    assert config.speculative_config is None
    ModeModelState(config, model, None, torch.device("cpu"))
    drafted = copy.copy(config)
    object.__setattr__(drafted, "speculative_config", SimpleNamespace(num_speculative_tokens=2))
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        ModeModelState(drafted, model, None, torch.device("cpu"))


def test_set_mlp_impl_refuses_a_compiled_engine(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from serving.vllm_plugin.state import set_mlp_impl

    config, model, _ = vllm_model
    eager = SimpleNamespace(model_runner=SimpleNamespace(vllm_config=config, model=model))
    assert set_mlp_impl(eager, "select") == 2
    for mode, graphs in (("VLLM_COMPILE", "NONE"), ("NONE", "FULL")):
        compilation = SimpleNamespace(
            mode=SimpleNamespace(name=mode), cudagraph_mode=SimpleNamespace(name=graphs)
        )
        compiled = SimpleNamespace(
            model_runner=SimpleNamespace(
                vllm_config=SimpleNamespace(compilation_config=compilation), model=model
            )
        )
        with pytest.raises(RuntimeError, match="change nothing"):
            set_mlp_impl(compiled, "triton")
    with pytest.raises(ValueError, match="must be one of"):
        set_mlp_impl(eager, "grouped")
    mlps = [m for m in model.modules() if hasattr(m, "stack_weights")]
    for mlp in mlps:
        monkeypatch.setattr(mlp, "tensor_parallel", 2)
    with pytest.raises(ValueError, match="tensor-parallel"):
        set_mlp_impl(eager, "triton")
    assert set_mlp_impl(eager, "loop") == 2
    assert set_mlp_impl(eager, "select") == 2
    assert {m.impl for m in mlps} == {"select"}


def test_impl_problem_names_what_each_impl_cannot_serve() -> None:
    from serving.vllm_plugin.model import impl_problem

    assert impl_problem("select", quantized=True, tensor_parallel=4) is None
    assert impl_problem("loop", quantized=False, tensor_parallel=2) is None
    assert impl_problem("triton", quantized=False, tensor_parallel=1) is None
    assert "must be one of" in (impl_problem("grouped", False, 1) or "")
    assert "unquantized" in (impl_problem("triton", True, 1) or "")
    assert "tensor-parallel" in (impl_problem("triton", False, 2) or "")


@pytest.mark.parametrize("impl", IMPLS)
def test_hand_driven_engine_matches_hf_greedy(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]],
    tiny: tuple[Qwen3ForCausalLM, Routing],
    impl: str,
) -> None:
    config, model, _ = vllm_model
    set_impl(model, impl)
    want_tokens, want_modes = tiny_greedy(*tiny, TINY_PROMPTS, NEW_TOKENS)
    engine = MiniEngine(config, model)
    req_ids = engine.run(TINY_PROMPTS)
    assert [engine.done[r] for r in req_ids] == want_tokens
    traces = engine.state.slots.traces()
    assert [traces[r] for r in req_ids] == want_modes
    assert engine.state.mixed_steps > 0
    assert {int(Mode.THINK), int(Mode.CODE), int(Mode.ANSWER)} <= {
        m for row in want_modes for m in row
    }


def test_hand_driven_engine_with_chunked_prefill_and_prefix_reuse(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]], tiny: tuple[Qwen3ForCausalLM, Routing]
) -> None:
    config, model, _ = vllm_model
    set_impl(model, "select")
    want_tokens, want_modes = tiny_greedy(*tiny, LONG_PROMPTS, NEW_TOKENS)
    engine = MiniEngine(config, model)
    first = engine.run(LONG_PROMPTS)
    second = engine.run(LONG_PROMPTS)
    for req_ids in (first, second):
        assert [engine.done[r] for r in req_ids] == want_tokens
        traces = engine.state.slots.traces()
        assert [traces[r] for r in req_ids] == want_modes
    assert min(engine.state.slots.first_starts[r] for r in second) >= 64
    assert {0, 1, 2} <= set(want_modes[0])


def test_hand_driven_engine_survives_preemption(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]], tiny: tuple[Qwen3ForCausalLM, Routing]
) -> None:
    config, model, _ = vllm_model
    set_impl(model, "loop")
    want_tokens, want_modes = tiny_greedy(*tiny, TINY_PROMPTS, NEW_TOKENS)
    engine = MiniEngine(config, model)
    req_ids = [engine.admit(p, NEW_TOKENS) for p in TINY_PROMPTS]
    for _ in range(6):
        engine.step()
    victim = req_ids[2]
    tokens, remaining = engine.preempt(victim)
    assert len(tokens) > len(TINY_PROMPTS[2])
    for _ in range(3):
        engine.step()
    engine.admit(tokens, NEW_TOKENS, req_id=victim)
    engine.running[-1].prompt_len = len(TINY_PROMPTS[2])
    assert remaining < NEW_TOKENS
    while engine.running:
        engine.step()
    assert [engine.done[r] for r in req_ids] == want_tokens
    traces = engine.state.slots.traces()
    assert [traces[r] for r in req_ids] == want_modes


@pytest.mark.parametrize("impl", ["select", IMPLS[2]])
def test_hand_driven_engine_under_torch_compile(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]],
    tiny: tuple[Qwen3ForCausalLM, Routing],
    exported: Path,
    impl: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.config import CompilationMode, set_current_vllm_config

    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    config = make_config(exported, enforce_eager=False, impl=impl)
    assert config.compilation_config.mode != CompilationMode.NONE
    want_tokens, want_modes = tiny_greedy(*tiny, TINY_PROMPTS, NEW_TOKENS)
    with set_current_vllm_config(config):
        model, _ = load_model(config, exported)
        engine = MiniEngine(config, model)
        for _ in range(2):
            req_ids = engine.run(TINY_PROMPTS)
            assert [engine.done[r] for r in req_ids] == want_tokens
            traces = engine.state.slots.traces()
            assert [traces[r] for r in req_ids] == want_modes


FRESH_PROCESS = """
import sys
from serving.vllm_plugin import register
from vllm.engine.arg_utils import EngineArgs
register()
for path in sys.argv[1:]:
    config = EngineArgs(model=path, skip_tokenizer_init=True, dtype="float32").create_model_config()
    print("RESULT", config.architectures[0], config.hf_config.model_type)
"""


def test_pilot_checkpoints_load_in_a_fresh_vllm_process(tmp_path: Path) -> None:
    trained = tiny_pilot_checkpoint("full", tmp_path / "trained")
    built = load_checkpoint(trained, torch.float32)
    exported = export_for_vllm(built.model, tmp_path / "export", token_texts=tiny_token_texts())
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root), "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
    run = subprocess.run(
        [sys.executable, "-c", FRESH_PROCESS, str(exported), str(trained)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=root,
        env=env,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    results = [line.split()[1:] for line in run.stdout.splitlines() if line.startswith("RESULT ")]
    assert results == [
        ["ChainedQwen3ForCausalLM", "qwen3"],
        ["ChainedQwen3ForCausalLM", "chained_qwen3"],
    ]


@pytest.mark.parametrize(
    ("variant", "export"),
    [("full", True), ("lora", True), ("full", False)],
    ids=["full-exported", "lora-exported", "full-as-saved"],
)
def test_pilot_checkpoints_serve_like_hf(
    vllm_model: tuple[Any, Any, dict[str, torch.Tensor]],
    variant: str,
    export: bool,
    tmp_path: Path,
) -> None:
    from vllm.config import set_current_vllm_config

    trained = tiny_pilot_checkpoint(variant, tmp_path / "trained")
    built = load_checkpoint(trained, torch.float32)
    assert built.routing is not None
    if export:
        served = export_for_vllm(built.model, tmp_path / "export", token_texts=tiny_token_texts())
    else:
        served = trained
        (served / TOKEN_TEXTS_FILE).write_text(json.dumps(tiny_token_texts()) + "\n")
    config = make_config(served, enforce_eager=True, impl="select")
    with torch.no_grad(), set_current_vllm_config(config):
        model, _ = load_model(config, served)
        engine = MiniEngine(config, model)
        req_ids = engine.run(TINY_PROMPTS)
        want_tokens, want_modes = tiny_greedy(built.model, built.routing, TINY_PROMPTS, NEW_TOKENS)
    assert [engine.done[r] for r in req_ids] == want_tokens
    traces = engine.state.slots.traces()
    assert [traces[r] for r in req_ids] == want_modes
    assert len({m for row in want_modes for m in row}) == 3

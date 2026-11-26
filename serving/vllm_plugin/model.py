from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM, Qwen3MLP
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors

from serving.export import MLP_IMPLS, tracking_settings
from serving.mixed import StackedWeights


@torch.library.custom_op("chained_transformer::grouped_mode_mlp", mutates_args=())
def grouped_mode_mlp_op(
    x: torch.Tensor, mode_ids: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    from serving.triton_grouped import grouped_mode_mlp

    return grouped_mode_mlp(x, mode_ids, StackedWeights(gate_up, down))


@grouped_mode_mlp_op.register_fake
def _grouped_mode_mlp_fake(
    x: torch.Tensor, mode_ids: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(x)


def impl_problem(impl: str, quantized: bool, tensor_parallel: int) -> str | None:
    if impl not in MLP_IMPLS:
        return f"mlp_impl must be one of {MLP_IMPLS}, got {impl!r}"
    if impl != "triton":
        return None
    if quantized:
        return (
            "mlp_impl 'triton' reads unquantized stacked weights; serve a quantized "
            "checkpoint with mlp_impl 'select'"
        )
    if tensor_parallel > 1:
        return (
            "mlp_impl 'triton' does not all-reduce across tensor-parallel ranks; serve on one "
            "GPU or with mlp_impl 'select'"
        )
    return None


class VllmModeMLP(nn.Module):
    def __init__(
        self, config: Any, quant_config: Any, prefix: str, num_modes: int, impl: str
    ) -> None:
        super().__init__()
        self.quantized = quant_config is not None
        self.tensor_parallel = get_tensor_model_parallel_world_size()
        problem = impl_problem(impl, self.quantized, self.tensor_parallel)
        if problem is not None:
            raise NotImplementedError(problem)
        self.modes = nn.ModuleList(
            Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.modes.{m}",
            )
            for m in range(num_modes)
        )
        self.impl = impl
        self._stacked: StackedWeights | None = None

    def stack_weights(self) -> StackedWeights:
        with torch.no_grad():
            gate_up = torch.stack([m.get_parameter("gate_up_proj.weight") for m in self.modes])
            down = torch.stack([m.get_parameter("down_proj.weight") for m in self.modes])
            for i, m in enumerate(self.modes):
                m.get_parameter("gate_up_proj.weight").data = gate_up[i]
                m.get_parameter("down_proj.weight").data = down[i]
        self._stacked = StackedWeights(gate_up, down)
        return self._stacked

    def stacking_problems(self) -> list[str]:
        if self._stacked is None:
            return ["mode weights were not stacked after loading"]
        problems = []
        for i, m in enumerate(self.modes):
            for name, whole in (
                ("gate_up_proj.weight", self._stacked.gate_up),
                ("down_proj.weight", self._stacked.down),
            ):
                weight = m.get_parameter(name)
                if weight.numel() == 0:
                    continue
                if weight.data_ptr() != whole[i].data_ptr():
                    problems.append(
                        f"mode {i} {name} no longer shares memory with the stacked weights; "
                        "vLLM replaced it after loading, so the triton impl would use stale "
                        "weights"
                    )
                elif weight.device != whole.device or weight.dtype != whole.dtype:
                    problems.append(f"mode {i} {name} is {weight.dtype} on {weight.device}")
        return problems

    def stacked(self) -> StackedWeights:
        if self._stacked is None:
            raise RuntimeError(
                "mode weights were not stacked after loading: "
                "ChainedQwen3ForCausalLM.load_weights stacks them, so the model was loaded "
                "some other way"
            )
        return self._stacked

    def forward(self, x: torch.Tensor, mode_ids: torch.Tensor | None) -> torch.Tensor:
        if mode_ids is None:
            return self.modes[0](x)
        ids = mode_ids[: x.shape[0]]
        if self.impl == "select":
            out = self.modes[0](x)
            for m in range(1, len(self.modes)):
                out = torch.where((ids == m).unsqueeze(-1), self.modes[m](x), out)
            return out
        if self.impl == "loop":
            out = torch.zeros_like(x)
            for m, mlp in enumerate(self.modes):
                rows = (ids == m).nonzero(as_tuple=True)[0]
                if rows.numel():
                    out.index_copy_(0, rows, mlp(x.index_select(0, rows)))
            return out
        w = self.stacked()
        return grouped_mode_mlp_op(x, ids, w.gate_up, w.down)


class ModeDecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config: Any,
        cache_config: Any = None,
        quant_config: Any = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            **kwargs,
        )
        settings = tracking_settings(config)
        self.mlp = VllmModeMLP(
            config, quant_config, f"{prefix}.mlp", settings.num_modes, settings.mlp_impl
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        mode_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, mode_ids)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "mode_ids": 0,
    }
)
class ChainedQwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=ModeDecoderLayer
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        mode_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if mode_ids is None:
            raise RuntimeError(
                "ChainedQwen3ForCausalLM got no mode_ids. vLLM passes them only through "
                "ModeModelState, which Model Runner V2 picks up from get_model_state_cls; "
                "this runner did not. Use vLLM 0.34 with its default GPU runner."
            )
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual, mode_ids)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class ChainedQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = ChainedQwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        for module in self.modules():
            if isinstance(module, VllmModeMLP):
                module.stack_weights()
        return loaded

    @staticmethod
    def get_model_state_cls() -> type:
        from serving.vllm_plugin.state import ModeModelState

        return ModeModelState

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        mode_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds, mode_ids)

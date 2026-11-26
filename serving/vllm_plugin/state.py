from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from chained_transformer.generate import TokenTexts
from serving.export import load_token_texts, tracking_settings
from serving.mode_slots import SlotModes


def token_texts(vllm_config: VllmConfig) -> list[str]:
    model_config = vllm_config.model_config
    settings = tracking_settings(model_config.hf_config)
    texts = load_token_texts(model_config.model, settings)
    if texts is not None:
        return texts
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer,
        revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
    )
    return TokenTexts.from_tokenizer(tokenizer).texts


class ModeModelState(DefaultModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: Any,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        settings = tracking_settings(vllm_config.model_config.hf_config)
        self.slots = SlotModes(token_texts(vllm_config), settings.prompt_mode, settings.record)
        self.mode_ids = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
        self.host_mode_ids = torch.zeros(self.max_num_tokens, dtype=torch.int64)
        self.mixed_steps = 0
        self.steps = 0

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        self.slots.add(req_index, new_req_data.req_id, new_req_data.prefill_token_ids)

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self.slots.remove(req_id)

    def prepare_inputs(self, input_batch: Any, req_states: Any) -> dict[str, Any]:
        inputs = dict(super().prepare_inputs(input_batch, req_states))
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        padded = input_batch.num_tokens_after_padding
        token_ids = input_batch.input_ids[:num_tokens].to("cpu").numpy()
        modes = self.slots.batch_modes(
            input_batch.req_ids[:num_reqs],
            np.asarray(input_batch.idx_mapping_np[:num_reqs]),
            np.asarray(input_batch.num_computed_tokens_np[:num_reqs]),
            np.asarray(input_batch.query_start_loc_np[: num_reqs + 1]),
            token_ids,
        )
        host = self.host_mode_ids[:padded]
        host.zero_()
        host[:num_tokens] = torch.from_numpy(modes)
        self.mode_ids[:padded].copy_(host)
        qsl = input_batch.query_start_loc_np
        real = [
            modes[qsl[row] : qsl[row + 1]]
            for row in range(num_reqs)
            if self.slots.owns(int(input_batch.idx_mapping_np[row]), input_batch.req_ids[row])
        ]
        if real:
            self.steps += 1
            self.mixed_steps += int(len(set(np.concatenate(real).tolist())) > 1)
        inputs["mode_ids"] = self.mode_ids[:padded]
        return inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        inputs = dict(super().prepare_dummy_inputs(num_reqs, num_tokens))
        self.mode_ids[:num_tokens].zero_()
        inputs["mode_ids"] = self.mode_ids[:num_tokens]
        return inputs


def _runner(worker: Any) -> Any:
    inner = getattr(worker, "worker", None) or worker
    return inner.model_runner


def set_mlp_impl(worker: Any, impl: str) -> int:
    from serving.vllm_plugin.model import VllmModeMLP

    mlps = [m for m in _runner(worker).model.modules() if isinstance(m, VllmModeMLP)]
    for mlp in mlps:
        mlp.impl = impl
    return len(mlps)


def collect_traces(worker: Any) -> dict[str, Any]:
    state = _runner(worker).model_state
    return {
        "traces": state.slots.traces(),
        "first_starts": dict(state.slots.first_starts),
        "steps": state.steps,
        "mixed_steps": state.mixed_steps,
        "rebuilds": sum(s.rebuilds for s in state.slots.slots.values()),
    }

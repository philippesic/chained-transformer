from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel

from chained_transformer.surgery import Routing, decoder_layers
from serving.mixed import IMPLS, StackedWeights, stack_any


@dataclass
class PagerStats:
    loads: int = 0
    evictions: int = 0
    bytes_moved: int = 0
    hits: int = 0
    streamed_bytes: int = 0
    over_capacity_steps: int = 0


class ModePager:
    def __init__(
        self, layers: Sequence[StackedWeights], capacity: int, device: str | torch.device
    ) -> None:
        if not layers:
            raise ValueError("need at least one layer")
        self.num_modes = layers[0].num_modes
        if not 1 <= capacity <= self.num_modes:
            raise ValueError(f"capacity must be in 1..{self.num_modes}, got {capacity}")
        self.device = torch.device(device)
        self.capacity = capacity
        pin = self.device.type == "cuda"
        self.host = [
            StackedWeights(
                w.gate_up.detach().to("cpu").contiguous(), w.down.detach().to("cpu").contiguous()
            )
            for w in layers
        ]
        if pin:
            self.host = [
                StackedWeights(w.gate_up.pin_memory(), w.down.pin_memory()) for w in self.host
            ]
        self.resident = [
            StackedWeights(
                torch.empty(
                    (capacity, *w.gate_up.shape[1:]), dtype=w.gate_up.dtype, device=self.device
                ),
                torch.empty((capacity, *w.down.shape[1:]), dtype=w.down.dtype, device=self.device),
            )
            for w in self.host
        ]
        self.slot_table = torch.full((self.num_modes,), -1, dtype=torch.long, device=self.device)
        self.slots: OrderedDict[int, int] = OrderedDict()
        self.stream = torch.cuda.Stream(self.device) if pin else None
        self.pending: dict[int, torch.cuda.Event] = {}
        self.scratch = [
            StackedWeights(
                torch.empty((1, *w.gate_up.shape[1:]), dtype=w.gate_up.dtype, device=self.device),
                torch.empty((1, *w.down.shape[1:]), dtype=w.down.dtype, device=self.device),
            )
            for w in self.host
        ]
        self.streamed: list[int] = []
        self.stats = PagerStats()

    def mode_bytes(self) -> int:
        return sum(
            w.gate_up[0].numel() * w.gate_up.element_size()
            + w.down[0].numel() * w.down.element_size()
            for w in self.host
        )

    def _free_slot(self, keep: set[int]) -> int:
        used = set(self.slots.values())
        for slot in range(self.capacity):
            if slot not in used:
                return slot
        for mode in self.slots:
            if mode not in keep:
                slot = self.slots.pop(mode)
                self.slot_table[mode] = -1
                self.pending.pop(mode, None)
                self.stats.evictions += 1
                return slot
        raise RuntimeError(
            f"{len(keep)} modes are needed at once but only {self.capacity} fit on the device"
        )

    def _load(self, mode: int, keep: set[int]) -> None:
        slot = self._free_slot(keep)
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                self._copy(mode, slot, non_blocking=True)
                event = torch.cuda.Event()
                event.record(self.stream)
            self.pending[mode] = event
        else:
            self._copy(mode, slot, non_blocking=False)
        self.slots[mode] = slot
        self.slot_table[mode] = slot
        self.stats.loads += 1
        self.stats.bytes_moved += self.mode_bytes()

    def _copy(self, mode: int, slot: int, non_blocking: bool) -> None:
        for host, dev in zip(self.host, self.resident, strict=True):
            dev.gate_up[slot].copy_(host.gate_up[mode], non_blocking=non_blocking)
            dev.down[slot].copy_(host.down[mode], non_blocking=non_blocking)

    def prefetch(self, mode: int, keep: Iterable[int] = ()) -> bool:
        if mode in self.slots:
            return True
        keep_set = set(keep) | {mode}
        if len(self.slots) >= self.capacity and all(m in keep_set for m in self.slots):
            return False
        self._load(mode, keep_set)
        return True

    def ensure(self, modes: Iterable[int]) -> None:
        need = set(int(m) for m in modes)
        if len(need) > self.capacity:
            raise RuntimeError(
                f"{len(need)} modes are needed at once but only {self.capacity} fit on the device"
            )
        for mode in sorted(need):
            if mode in self.slots:
                self.stats.hits += 1
                self.slots.move_to_end(mode)
            else:
                self._load(mode, need)
        for mode in need:
            event = self.pending.pop(mode, None)
            if event is not None:
                torch.cuda.current_stream(self.device).wait_event(event)

    def begin_step(self, mode_ids: torch.Tensor) -> list[int]:
        counts = torch.bincount(mode_ids.reshape(-1).cpu(), minlength=self.num_modes).tolist()
        needed = sorted((m for m in range(self.num_modes) if counts[m]), key=lambda m: -counts[m])
        keep = needed[: self.capacity]
        self.ensure(keep)
        self.streamed = needed[self.capacity :]
        if self.streamed:
            self.stats.over_capacity_steps += 1
        return self.streamed

    def stream_layer(self, layer: int, mode: int) -> StackedWeights:
        host, buf = self.host[layer], self.scratch[layer]
        buf.gate_up[0].copy_(host.gate_up[mode])
        buf.down[0].copy_(host.down[mode])
        self.stats.streamed_bytes += (
            host.gate_up[0].numel() * host.gate_up.element_size()
            + host.down[0].numel() * host.down.element_size()
        )
        return buf

    def weights(self, layer: int) -> StackedWeights:
        return self.resident[layer]

    def remap(self, mode_ids: torch.Tensor) -> torch.Tensor:
        return self.slot_table[mode_ids]


class PagedModeMLP(nn.Module):
    def __init__(
        self, pager: ModePager, layer: int, routing: Routing, impl: str = "grouped"
    ) -> None:
        super().__init__()
        self.pager = pager
        self.layer = layer
        self.routing = routing
        self.impl = IMPLS[impl]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        ids = self.routing.flat_ids(flat.shape[0]).to(flat.device)
        if self.layer == 0:
            self.pager.begin_step(ids)
        slots = self.pager.remap(ids)
        if not self.pager.streamed:
            return self.impl(flat, slots, self.pager.weights(self.layer)).reshape(shape)
        out = torch.zeros_like(flat)
        rows = (slots >= 0).nonzero(as_tuple=True)[0]
        if rows.numel():
            part = self.impl(flat[rows], slots[rows], self.pager.weights(self.layer))
            out.index_copy_(0, rows, part)
        for mode in self.pager.streamed:
            rows = (ids == mode).nonzero(as_tuple=True)[0]
            w = self.pager.stream_layer(self.layer, mode)
            out.index_copy_(0, rows, self.impl(flat[rows], torch.zeros_like(rows), w))
        return out.reshape(shape)


def use_paged_mlps(
    model: PreTrainedModel,
    routing: Routing,
    capacity: int,
    device: str | torch.device,
    impl: str = "grouped",
) -> ModePager:
    layers = list(decoder_layers(model))
    pager = ModePager([stack_any(layer.get_submodule("mlp")) for layer in layers], capacity, device)
    for i, layer in enumerate(layers):
        layer.mlp = PagedModeMLP(pager, i, routing, impl)
    return pager

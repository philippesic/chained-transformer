from __future__ import annotations

import torch

from tools.memtrace import LiveBytes


def test_live_bytes_tracks_allocations_frees_and_phase_peaks() -> None:
    tracker = LiveBytes()
    with tracker:
        tracker.set_phase("a")
        x = torch.empty(1 << 20, dtype=torch.uint8)
        y = torch.empty(2 << 20, dtype=torch.uint8)
        del y
        tracker.set_phase("b")
        z = torch.empty(512 << 10, dtype=torch.uint8)
        view = x[:10]
    assert tracker.peaks["a"].bytes == 3 << 20
    assert tracker.peaks["b"].bytes == (1 << 20) + (512 << 10)
    assert view.untyped_storage().nbytes() == 1 << 20
    del x, z, view
    assert tracker.current == 0

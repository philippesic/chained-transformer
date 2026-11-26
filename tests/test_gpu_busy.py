from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

import pytest

from tools import gpu_smoke, pilot

DESKTOP = "2817, /usr/libexec/xdg-desktop-portal-gnome, 5 MiB"
JOB = "4242, python, 12000 MiB"
UNKNOWN = "77, /usr/bin/gnome-shell, [N/A]"


def _fake_smi(lines: list[str]) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[0] == "nvidia-smi"
        return subprocess.CompletedProcess(cmd, 0, stdout="".join(f"{x}\n" for x in lines))

    return run


def test_gpu_jobs_skip_small_clients_and_keep_unreadable_ones() -> None:
    assert gpu_smoke.gpu_jobs([DESKTOP, "9, Xorg, 63 MiB"]) == []
    assert gpu_smoke.gpu_jobs(["9, python, 64 MiB", JOB, UNKNOWN, "garbled"]) == [
        "9, python, 64 MiB",
        JOB,
        UNKNOWN,
        "garbled",
    ]


@pytest.mark.parametrize("busy", [gpu_smoke.other_gpu_clients, pilot.other_gpu_processes])
def test_busy_checks_ignore_desktop_processes_only(
    monkeypatch: pytest.MonkeyPatch, busy: Callable[[], list[str] | None]
) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_smi([DESKTOP]))
    assert busy() == []
    monkeypatch.setattr(subprocess, "run", _fake_smi([DESKTOP, JOB, UNKNOWN]))
    assert busy() == [JOB, UNKNOWN]

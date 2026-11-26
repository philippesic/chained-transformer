from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("CHAINED_TRANSFORMER_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def register() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from serving.vllm_plugin import register as register_models

    register_models()

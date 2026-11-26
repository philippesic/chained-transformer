from __future__ import annotations

import os

import pytest
import torch

from serving.mixed import StackedWeights, mode_mlp_loop

if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("no GPU and TRITON_INTERPRET is not 1", allow_module_level=True)

triton_grouped = pytest.importorskip("serving.triton_grouped")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def weights(seed: int, modes: int, hidden: int, inter: int, dtype: torch.dtype) -> StackedWeights:
    g = torch.Generator().manual_seed(seed)
    w = StackedWeights(
        torch.randn(modes, 2 * inter, hidden, generator=g) * 0.1,
        torch.randn(modes, hidden, inter, generator=g) * 0.1,
    )
    return StackedWeights(w.gate_up.to(DEVICE, dtype), w.down.to(DEVICE, dtype))


def mode_ids(pattern: str, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ids = {
        "mixed": torch.randint(0, 3, (n,), generator=g),
        "uniform": torch.full((n,), 1),
        "one_empty": torch.randint(0, 2, (n,), generator=g) * 2,
        "sorted": torch.sort(torch.randint(0, 3, (n,), generator=g)).values,
    }[pattern]
    return ids.to(DEVICE)


@pytest.mark.parametrize("pattern", ["mixed", "uniform", "one_empty", "sorted"])
@pytest.mark.parametrize("num_tokens", [1, 5, 37])
def test_triton_mlp_matches_reference(pattern: str, num_tokens: int) -> None:
    w = weights(0, 3, 96, 80, torch.float32)
    x = torch.randn(num_tokens, 96, generator=torch.Generator().manual_seed(1)).to(DEVICE)
    ids = mode_ids(pattern, num_tokens, 2)
    want = mode_mlp_loop(x, ids, w)
    got = triton_grouped.grouped_mode_mlp(x, ids, w, block_m=16, block_n=32, block_k=32)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_triton_mlp_half_precision() -> None:
    w = weights(3, 3, 64, 64, torch.float16)
    x = (torch.randn(21, 64, generator=torch.Generator().manual_seed(4)) * 0.5).to(DEVICE)
    x = x.to(torch.float16)
    ids = mode_ids("mixed", 21, 5)
    want = mode_mlp_loop(x.float(), ids, StackedWeights(w.gate_up.float(), w.down.float()))
    got = triton_grouped.grouped_mode_mlp(x, ids, w, block_m=16, block_n=32, block_k=32)
    torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-3)


def test_grouped_matmul_matches_per_token_weights() -> None:
    g = torch.Generator().manual_seed(6)
    w = torch.randn(4, 48, 40, generator=g).to(DEVICE)
    x = torch.randn(29, 40, generator=g).to(DEVICE)
    ids = torch.randint(0, 4, (29,), generator=g).to(DEVICE)
    want = torch.bmm(w[ids], x.unsqueeze(-1)).squeeze(-1)
    got = triton_grouped.grouped_matmul(x, w, ids, block_m=16, block_n=16, block_k=16)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_float32_inputs_use_exact_dots() -> None:
    assert triton_grouped.dot_precision(torch.zeros(1, dtype=torch.float32)) == "ieee"
    assert triton_grouped.dot_precision(torch.zeros(1, dtype=torch.float16)) == "tf32"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("dtype", "tl.dot needs both the same dtype"),
        ("ids_len", "mode_ids must be \\[tokens\\]"),
        ("ids_float", "integer tensor"),
        ("features", "features but the weights expect"),
        ("block", "power of two of at least 16"),
    ],
)
def test_triton_mlp_rejects_bad_inputs(change: str, message: str) -> None:
    w = weights(7, 3, 64, 64, torch.float32)
    x = torch.randn(9, 64).to(DEVICE)
    ids = mode_ids("mixed", 9, 8)
    block = 16
    if change == "dtype":
        x = x.to(torch.float16)
    elif change == "ids_len":
        ids = ids[:5]
    elif change == "ids_float":
        ids = ids.float()
    elif change == "features":
        x = x[:, :32]
    else:
        block = 24
    with pytest.raises(ValueError, match=message):
        triton_grouped.grouped_mode_mlp(x, ids, w, block_m=16, block_n=block, block_k=32)

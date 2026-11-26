from __future__ import annotations

import pytest
import torch

from chained_transformer.optim import (
    BLOCK,
    BlockQuantizer,
    ChunkedAdamW8bit,
    dynamic_map,
    fp32_to_bf16_stochastic,
)


@pytest.mark.parametrize("signed", [True, False])
def test_dynamic_map_is_sorted_and_spans_unit_range(signed: bool) -> None:
    qmap = dynamic_map(signed)
    assert qmap.numel() == 256
    assert torch.all(qmap[1:] >= qmap[:-1])
    assert float(qmap.max()) == 1.0 and 0.0 in qmap.tolist()
    assert float(qmap.min()) < 0 if signed else float(qmap.min()) == 0.0


@pytest.mark.parametrize("signed", [True, False])
def test_block_quantization_round_trip_is_close(signed: bool) -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(BLOCK * 8, generator=g)
    if not signed:
        x = x.square()
    q = BlockQuantizer(signed, torch.device("cpu"))
    codes, scale = q.quantize(x)
    back = q.dequantize(codes, scale)
    assert codes.dtype == torch.uint8 and scale.numel() == 8
    rel = (back - x).abs() / scale.repeat_interleave(BLOCK)
    assert float(rel.max()) < 0.05


def test_stochastic_rounding_is_unbiased_and_exact_on_bf16_values() -> None:
    exact = torch.tensor([1.0, -2.5, 0.15625])
    assert torch.equal(fp32_to_bf16_stochastic(exact).float(), exact)
    x = torch.full((200_000,), 1.0 + 2**-10)
    g = torch.Generator().manual_seed(1)
    mean = float(fp32_to_bf16_stochastic(x, g).float().mean())
    assert abs(mean - (1.0 + 2**-10)) < 2e-5
    assert float(x.bfloat16().float().mean()) == 1.0


def _regression(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(64, 32, generator=g)
    w = torch.randn(32, 16, generator=g)
    return x, x @ w * 0.1


def _run(opt_factory: object, steps: int, dtype: torch.dtype = torch.float32) -> list[float]:
    torch.manual_seed(0)
    layer = torch.nn.Linear(32, 16 * 512, bias=False).to(dtype)
    head = torch.eye(16, dtype=dtype).repeat_interleave(512, 0) / 512
    opt = opt_factory(layer.parameters())  # type: ignore[operator]
    x, y = _regression(0)
    losses = []
    for _ in range(steps):
        pred = (layer(x.to(dtype)) @ head).float()
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
    return losses


def test_chunked_adamw8bit_tracks_fp32_adamw() -> None:
    ref = _run(lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=0.0), 40)
    got = _run(lambda p: ChunkedAdamW8bit(p, lr=1e-2, chunk_numel=BLOCK * 64), 40)
    assert got[-1] < 0.5 * got[0]
    assert abs(got[-1] - ref[-1]) / ref[0] < 0.05


def test_chunk_size_does_not_change_the_result() -> None:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        a = _run(lambda p: ChunkedAdamW8bit(p, lr=1e-2, chunk_numel=BLOCK), 5)
        b = _run(lambda p: ChunkedAdamW8bit(p, lr=1e-2, chunk_numel=BLOCK * 4096), 5)
    finally:
        torch.set_num_threads(threads)
    assert a == b


def test_bf16_params_train_with_stochastic_rounding() -> None:
    losses = _run(lambda p: ChunkedAdamW8bit(p, lr=1e-2), 30, torch.bfloat16)
    assert losses[-1] < 0.5 * losses[0]


def test_small_params_keep_fp32_state_and_state_dict_round_trips() -> None:
    p = torch.nn.Parameter(torch.ones(10))
    big = torch.nn.Parameter(torch.ones(BLOCK * 16))
    opt = ChunkedAdamW8bit([p, big], lr=0.1)
    (p.sum() + big.sum()).backward()
    opt.step()
    assert opt.state[p]["m"].dtype == torch.float32
    assert opt.state[big]["m_codes"].dtype == torch.uint8
    clone = ChunkedAdamW8bit([p, big], lr=0.1)
    clone.load_state_dict(opt.state_dict())
    assert torch.equal(clone.state[big]["m_codes"], opt.state[big]["m_codes"])
    assert clone.state[big]["m_codes"].dtype == torch.uint8
    assert clone.state[big]["m_scale"].dtype == torch.float32
    assert clone.state[p]["v"].dtype == torch.float32
    assert clone.state[big]["step"] == 1


def test_state_dict_round_trip_keeps_state_dtypes_for_bf16_params() -> None:
    big = torch.nn.Parameter(torch.ones(BLOCK * 16, dtype=torch.bfloat16))
    small = torch.nn.Parameter(torch.ones(10, dtype=torch.bfloat16))
    opt = ChunkedAdamW8bit([big, small], lr=0.1)
    (big.float().sum() + small.float().sum()).backward()
    opt.step()
    clone = ChunkedAdamW8bit([big, small], lr=0.1)
    clone.load_state_dict(opt.state_dict())
    assert clone.state[big]["v_codes"].dtype == torch.uint8
    assert clone.state[big]["v_scale"].dtype == torch.float32
    assert clone.state[small]["m"].dtype == torch.float32


def test_fp32_grads_are_left_untouched_by_the_step() -> None:
    p = torch.nn.Parameter(torch.ones(BLOCK * 16))
    p.grad = torch.full_like(p, 3.0)
    ChunkedAdamW8bit([p], lr=0.1).step()
    assert torch.equal(p.grad, torch.full_like(p, 3.0))


@pytest.mark.parametrize("numel", [BLOCK * 16, 1000, 4100])
def test_uneven_last_chunk_matches_a_single_chunk(numel: int) -> None:
    g = torch.Generator().manual_seed(3)
    init = torch.randn(numel, generator=g)
    grads = [torch.randn(numel, generator=g) for _ in range(3)]
    results = []
    for chunk in (BLOCK * 3, BLOCK * 4096):
        p = torch.nn.Parameter(init.clone())
        opt = ChunkedAdamW8bit([p], lr=1e-2, weight_decay=0.1, chunk_numel=chunk)
        for grad in grads:
            p.grad = grad.clone()
            opt.step()
        results.append((p.detach().clone(), opt.state[p]))
    (pa, sa), (pb, sb) = results
    assert ("m_codes" in sa) == (numel == BLOCK * 16)
    assert torch.equal(pa, pb)
    for key, value in sa.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, sb[key]), key


def test_chunk_size_must_be_a_whole_number_of_blocks() -> None:
    with pytest.raises(ValueError, match="multiple"):
        ChunkedAdamW8bit([torch.nn.Parameter(torch.ones(4))], chunk_numel=BLOCK + 1)

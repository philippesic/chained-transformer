# Serving notes

Work log for the serving path ([design](serving.md)): what is verified, where, and what is not.
Rewrite rather than append.

## Environment of the 2026-11-26 overnight cloud run

- CPU-only container: 4 Xeon cores (AVX-512, AMX), 15 GB RAM, no GPU.
- huggingface.co, ModelScope and download.pytorch.org were unreachable, so there were **no
  Qwen3-0.6B weights or tokenizer**. Every check used tiny random Qwen3 models. The repo's two
  tokenizer tests in `tests/test_data.py` fail there for that reason only.
- PyPI and `git clone` from GitHub worked.

## Verified

## Not verified

## Blocked or skipped

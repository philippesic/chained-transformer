from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from chained_transformer.checkpoint import convert, save_checkpoint
from chained_transformer.modes import THINK_CLOSE, THINK_OPEN
from chained_transformer.surgery import SurgeryConfig, Variant

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", THINK_OPEN, THINK_CLOSE]
KEPT_ON_DECODE = [THINK_OPEN, THINK_CLOSE]
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
    "{% if enable_thinking is defined and enable_thinking is false %}"
    "{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}"
)
CORPUS = [
    "Let me think step by step. Wait, actually the answer is 4.",
    "Please reason step by step, and put your final answer within \\boxed{}.",
    "```python\ndef add(a, b):\n    return a + b\n```\n",
    "The final answer is \\boxed{12}. Hmm, alternatively we could check it.",
    "import sys\nfor line in sys.stdin:\n    print(sum(map(int, line.split())))\n",
    "class Solution:\n    def twoSum(self, nums, target):\n        return [0, 1]\n",
] * 20


def toy_tokenizer(vocab_size: int = 400) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer)
    tok.add_tokens([AddedToken(t, special=False, normalized=False) for t in KEPT_ON_DECODE])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        chat_template=CHAT_TEMPLATE,
    )


def tiny_base(vocab_size: int, seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
    return model


def bias_tokens(
    model: Qwen3ForCausalLM, tokenizer: PreTrainedTokenizerFast, strings: list[str], scale: float
) -> None:
    ids = {t for s in strings for t in tokenizer(s, add_special_tokens=False).input_ids}
    with torch.no_grad():
        for i in ids:
            model.lm_head.weight[i].mul_(scale)


def build(out: Path, variant: Variant | None, seed: int = 0) -> Path:
    tokenizer = toy_tokenizer()
    model = tiny_base(len(tokenizer), seed)
    bias_tokens(model, tokenizer, ["</think>", "```", "\n"], 2.5)
    if variant is None:
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out))
        tokenizer.save_pretrained(str(out))
        return out
    chained = convert(model, SurgeryConfig(variant=variant, lora_rank=4), "tiny-random", seed)
    with torch.no_grad():
        for name, p in chained.named_parameters():
            if ".modes.1." in name or ".modes.2." in name or "lora_b" in name:
                p.add_(0.05 * torch.randn_like(p))
    save_checkpoint(chained, out, tokenizer)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a tiny random checkpoint for plumbing runs")
    parser.add_argument("--variant", choices=["full", "lora", "moe", "plain"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    variant: Variant | None = None if args.variant == "plain" else args.variant
    build(args.out, variant, args.seed)
    print(json.dumps({"out": str(args.out), "variant": args.variant}))


if __name__ == "__main__":
    main()

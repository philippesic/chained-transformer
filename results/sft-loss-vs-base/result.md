# Pilot `sft-s0`: training lowers loss 0.16 nats below the base on unseen rows

**Conclusion.** The pilot's first run (`sft`, seed 0) is learning. On rows it had not seen, its
loss at the end of the first pass is 0.157 nats per token below the base model's on the same
rows (standard error 0.007, four steps), about 12% lower. On rows it had seen once it is 0.234
below (standard error 0.008). The logged loss curve looked flat because two thirds of the gain
arrived in the first 20 steps, during warmup, and row difficulty hides the rest: the base
model's loss on one step's eight rows ranges from 1.12 to 1.58.

Trained minus base, nats per token, on the eight rows each step trained on:

| Step | Pass | Base | Trained | All | think | answer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1.332 | 1.332 | +0.000 | +0.000 | -0.001 |
| 1 | 1 | 1.323 | 1.321 | -0.002 | -0.002 | -0.001 |
| 5 | 1 | 1.465 | 1.430 | -0.034 | -0.035 | -0.032 |
| 10 | 1 | 1.243 | 1.169 | -0.074 | -0.072 | -0.087 |
| 20 | 1 | 1.230 | 1.129 | -0.101 | -0.095 | -0.147 |
| 60 | 1 | 1.287 | 1.158 | -0.129 | -0.120 | -0.183 |
| 120 | 1 | 1.398 | 1.256 | -0.142 | -0.132 | -0.195 |
| 200 | 1 | 1.423 | 1.267 | -0.155 | -0.153 | -0.177 |
| 272 to 275 | 1 | 1.323 | 1.166 | -0.157 | -0.150 | -0.208 |
| 314 to 317 | 2 | 1.331 | 1.097 | -0.234 | -0.225 | -0.299 |

`answer` tokens gain most and `think` tokens follow. `code` tokens appear in 5 of the 16 steps,
at most 303 per step, and move by 0.00 to -0.03: there is too little code in this data to
train on. On 300 sampled rows the loss tokens are 86.7% `think`, 12.9% `answer` and 0.39%
`code`, and 89% of rows have no code tokens, so the `full` arm's code MLPs will stay near the
base.

**Conditions.** Run `artifacts/pilot/pilot-0p6b-8k/sft-s0` on the RTX 2080 Ti (branch
`cloud/moe-fix` at 2a66137; learning rate 1e-5 with 50 warmup steps, 8 rows per step, sequence
length 8,192, data `artifacts/data/ot3-natural-8k`), read while it was at about step 350 of
1,000. `tools/loss_vs_base.py` recomputes the base Qwen3-0.6B loss in fp32 on CPU for the rows of
a logged step and subtracts it from the logged loss. Two checks: the rows it assigns to each
step reproduce the logged token count for all 342 steps logged at the time, and at steps 0 and 1,
where the learning rate is near zero, the difference is 0.000 and -0.002. A pass is 281 steps.

**Limitations.** One run and 16 of its steps, 128 rows. This is training loss on the training
distribution, not benchmark accuracy: it shows the fine-tune changes the model, not that it
helps MATH500 or the coding benchmarks. The trained loss is the logged bf16 GPU value and the
base loss an fp32 CPU value; the agreement at steps 0 and 1 bounds that difference at about
0.002.

**Disposition.** Retained: the pilot is worth its GPU time on this count. Evidence:
[steps.jsonl](steps.jsonl) and [summary.json](summary.json). Regenerate with
`python -m tools.loss_vs_base --run-dir artifacts/pilot/pilot-0p6b-8k/sft-s0 --steps 0 1 5 10 20 60 120 200 272-275 314-317 --calibration-below 2 --out OUT.jsonl`
(about 6 minutes per step on 4 CPU threads). The working copy in the overnight worktree,
`artifacts/detect/`, is disposable.

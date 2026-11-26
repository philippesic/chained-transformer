# Single-mode upgrade on a toy task: thinking stays exact, but answers break unless the loss covers them

**Conclusion.** On a tiny synthetic task, retraining only the `code` MLPs of a mode model left
every position before the first code token bit-identical (largest log-probability change 0.0 on
both seeds) and learned the new code rule perfectly. But with loss on code tokens only, the
frozen `answer` mode then gave answers that matched the model's own new code in only 1% and 5%
of greedy generations. The `answer` mode had learned to work out the answer from the old code
pattern rather than from the last code token, and nothing in a code-only loss repairs that.
Putting the loss on every token, while still training only the `code` MLPs, raised that to 78%
and 72%. A plain fine-tune of all parameters on whole new traces reached 100% but moved
thinking too (small KL, nonzero). Every baseline trained on code tokens alone damaged either
thinking or answers. So on the GPU, a code upgrade should be judged end to end, and its loss
should include the answer tokens that follow code.

Greedy generation on 256 held-out new-rule traces, seed 0 / seed 1, in percent:

| Upgrade | Changed params | Largest log-prob change before first code position | Think exact | Code exact | Answer matches own code | Whole trace exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Modes: code MLPs only, loss on code | 49,152 | 0 / 0 | 100 / 100 | 100 / 100 | 1 / 5 | 1 / 5 |
| Modes: code MLPs only, loss on all tokens | 49,152 | 0 / 0 | 100 / 100 | 99 / 99 | 78 / 72 | 78 / 72 |
| Modes: all parameters, loss on code | 102,784 | 10.5 / 16 | 100 / 64 | 100 / 91 | 0 / 5 | 0 / 3 |
| Plain: all MLPs, loss on code | 49,152 | 10.4 / 8.9 | 100 / 100 | 100 / 100 | 0 / 0 | 0 / 0 |
| Plain: all parameters, loss on code | 78,208 | 12.7 / 14 | 91 / 89 | 91 / 89 | 0 / 0 | 0 / 0 |
| Plain: all parameters, loss on all tokens | 78,208 | 5.2 / 4.7 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

Before any upgrade both models generated every old-rule trace exactly (100%) and no new-rule
code (0%). Teacher-forced `answer` accuracy on new-rule traces was already only 75% / 79% (mode
model) and 73% / 83% (plain model), which is how the shortcut shows. The code-only mode upgrade
left it there (76% / 76%); loss on all tokens raised it to 95% / 93%. Thinking's KL from the
parent was exactly 0 for both mode upgrades and between 0.00004 and 0.19 for the baselines that touched
shared or thinking weights.

**Conditions.** `python -m tools.upgrade_demo` at this commit, CPU, fp32, two seeds. Model: Qwen3
shape with hidden size 64, 2 layers, vocabulary 64; the mode model is a `full` conversion of the
same random base. Each 39-token row is a 4-token prompt, 16 `think` tokens counting from the first
prompt digit, a fence, 12 `code` tokens starting from the sum of two prompt digits and stepping
by 3 (old rule) or 5 (new rule) modulo 16, a fence, and 4 `answer` tokens computed from the last
code token. Test rows use prompt-digit pairs never seen in training. Pretraining: 1,500 steps of
32 rows on old-rule traces, all parameters, AdamW at 3e-3. Upgrades: 300 steps of 32 new-rule
rows at 1e-3. "Answer matches own code" checks the answer against the model's own last code
token, so it isolates the `answer` mode from code errors. Full numbers, including old-rule test
rows and teacher-forced loss per label, are in [report.json](report.json).

**Limitations.** A 2-layer toy with a hand-made rule is not evidence about Qwen3 or real code.
The task's answer depends on code in a way the model could shortcut; real `answer` text may
depend on code more loosely or more tightly. One upgrade length and learning rate were tried.
The first version of the task let old and new code end on disjoint tokens, so new-rule answers
were outside anything the answer mode had seen; that run is superseded and not reported.

**Disposition.** Shapes experiment 2 in [the paper outline](../../docs/paper-outline.md): judge a
code upgrade by end-to-end benchmark scores, not per-label loss, and include an arm that puts
the loss on whole responses. `chained_transformer.upgrade retrain --loss-on all` does that.

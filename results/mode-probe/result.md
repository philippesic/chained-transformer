# Mode probe: the base model's hidden states already separate think, code and answer by layer 2

**Conclusion.** A linear probe on base Qwen3-0.6B hidden states tells `think`, `code` and
`answer` positions apart with 80% balanced accuracy at layer 2 and about 90% from layer 12 on,
against 33% for the majority class. The mode is in the residual stream almost from the start,
before any mode-specific weights exist.

Balanced accuracy on held-out rows, three classes (layer 0 is the embedding):

| Layer | 0 | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Position mode | 0.545 | 0.728 | 0.800 | 0.862 | 0.852 | 0.895 | 0.893 | 0.908 | 0.890 | 0.899 |
| Segment label | 0.550 | 0.733 | 0.802 | 0.858 | 0.854 | 0.890 | 0.890 | 0.905 | 0.885 | 0.892 |

Two readings for the design. A learned per-token router has the signal it needs from the first
layers, so the `moe` arm can find the modes by itself and hard routing adds little
information it lacks. And for the looping note's option 3 (mode-specific ends, shared middle),
specialisation could start at the first layers; nothing here says the middle layers need
their own mode weights.

**Conditions.** Runbook step 8, on CPU on the 2080 Ti host:
`python -m tools.mode_probe --data artifacts/data/ot3-mix-8k --max-tokens 8192 --rows 60`
(fp32, 55 rows kept, 64 positions per label per row, 7,626 positions; every fifth row held out:
704 `think`, 247 `code` and 704 `answer` test positions). Evidence:
[mode-probe-8k.json](mode-probe-8k.json).

**Limitations.** One seed and 55 rows. The tool's default `--max-tokens 4096` keeps only rows
that fit, 12 here, and none of their held-out positions is code, because code sits at the end
of long rows; that run ([mode-probe-4k.json](mode-probe-4k.json)) is a two-class result (0.82
at layer 2, 0.98 at layer 20) and should not be read as three-class. Position mode and segment
label differ only at mode boundaries, so the two rows are nearly the same measurement. A probe
shows the information is present, not that the model uses it.

**Disposition.** Retained as context for the routing design; it does not change the pilot. The
probe outputs in the overnight worktree (`artifacts/probes`) are disposable.

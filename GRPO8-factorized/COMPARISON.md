# GRPO8 vs SetTransformer

Both seed-8 models use the same 8-class dataset, frozen DINOSAUR checkpoint,
train/validation/test split, and 800-image test set. SetTransformer ranking is
label-free predicted-greedy probability ranking.

## Overall

| metric | GRPO8 | SetTransformer | GRPO delta |
|---|---:|---:|---:|
| accuracy@3 | 92.63 | 92.88 | -0.25 |
| accuracy@4 | 93.00 | 92.75 | +0.25 |
| pair@3, threshold 0.2 | **88.38** | 82.00 | **+6.38** |
| pair@4, threshold 0.2 | **92.75** | 88.13 | **+4.63** |
| pair@3, threshold 0.4 | **76.63** | 69.75 | **+6.88** |
| pair@4, threshold 0.4 | **82.50** | 77.38 | **+5.13** |

Paired image-level bootstrap 95% confidence intervals for the four pair deltas
are respectively `[+3.50,+9.13]`, `[+2.25,+7.00]`, `[+3.50,+10.25]`, and
`[+2.13,+8.00]` percentage points.

## Pair delta by class (GRPO8 - SetTransformer)

| class | @3/.2 | @4/.2 | @3/.4 | @4/.4 |
|---|---:|---:|---:|---:|
| bed + person | +2 | +1 | -3 | -4 |
| bowl + dining table | +6 | +2 | +7 | +6 |
| horse + person | -8 | -8 | -10 | -10 |
| motorcycle + car | +25 | +20 | +22 | +17 |
| pizza + dining table | +1 | +1 | 0 | 0 |
| sandwich + dining table | -3 | -4 | -1 | -1 |
| truck + car | +26 | +21 | +41 | +33 |
| umbrella + person | +2 | +4 | -1 | 0 |

The overall gain comes primarily from evidence localization: at threshold 0.4,
GRPO improves evidence@3/@4 by `+7.88/+5.75` points, versus only
`+1.00/+0.88` for anchor localization. The largest gains are in the shared-car
group, while horse + person remains worse than SetTransformer.


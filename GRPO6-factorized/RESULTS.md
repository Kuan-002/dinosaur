# GRPO6 evidence-selection results

All bbox numbers below are evaluation-only. Training uses frozen DINOSAUR
slots. Increment, novelty, necessity, and balanced-margin use the atomic class
loss; factorized MIL additionally uses the two image-level components already
encoded by each pair-class label. It never uses a bbox, mask, or slot-object
assignment.

## Seed-8 ablation

| Method | test acc@3 | test acc@4 | pair@3, .2 | pair@4, .2 | pair@3, .4 | pair@4, .4 |
|---|---:|---:|---:|---:|---:|---:|
| Increment | 82.33 | 82.67 | 79.67 | 86.83 | 60.33 | 69.17 |
| Attention novelty | 81.50 | 82.33 | 76.33 | 85.17 | 57.00 | 67.83 |
| Leave-one-out necessity | 83.67 | 83.33 | 74.00 | 84.33 | 57.17 | 68.17 |
| Balanced confusable margin | 82.83 | 82.83 | 72.33 | 82.67 | 57.67 | 68.33 |
| Factorized MIL | **83.67** | **83.17** | **84.17** | **91.00** | **72.50** | **80.83** |

## Factorized MIL replication

| Seed | valid acc | valid mean slots | test acc@3 | test acc@4 | pair@3, .2 | pair@4, .2 | pair@3, .4 | pair@4, .4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 83.50 | 5.20 | 83.67 | 83.17 | 84.17 | 91.00 | 72.50 | 80.83 |
| 17 | 83.50 | 5.16 | 83.67 | 84.50 | 80.50 | 86.83 | 68.33 | 76.33 |
| Mean | **83.50** | **5.18** | **83.67** | **83.83** | **82.33** | **88.92** | **70.42** | **78.58** |

Relative to the increment baseline, the two-seed mean improves pair coverage
by +2.66/+2.09 points at threshold .2 (@3/@4) and +10.09/+9.41 points at
threshold .4, while also improving classification.

The improvement is therefore strongest for concentrated object coverage. The
remaining limitation is efficiency: the current confidence-plus-novelty stop
uses about 5.18 slots, although the forced @3 ranking is already stronger than
the baseline. Stop calibration should be treated as a separate ablation.

## Supervision boundary

Factorized MIL is not a localization Oracle: it has no spatial target. It is,
however, stronger supervision than treating the six labels as unrelated
integers because it uses the known decomposition of each label into two
image-level components. This is justified only when that decomposition is part
of the task label definition. If the final protocol requires strictly atomic
six-class labels, the novelty/necessity/balanced experiments show that the
evidence objective remains underdetermined and a data intervention is needed.

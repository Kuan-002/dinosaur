# GRPO8 factorized results

Run: `grpo8_factorized_joint_seed8_20260716`  
Best checkpoint: epoch 26 (`valid_pair_score=0.888457`)  
Test set: 800 images (100 per class)

## Overall bbox metrics

| metric | @3 | @4 |
|---|---:|---:|
| accuracy | 92.63 | 93.00 |
| anchor, threshold 0.2 | 94.25 | 96.88 |
| evidence, threshold 0.2 | 93.50 | 95.75 |
| pair, threshold 0.2 | **88.38** | **92.75** |
| anchor, threshold 0.4 | 87.50 | 90.75 |
| evidence, threshold 0.4 | 85.63 | 89.25 |
| pair, threshold 0.4 | **76.63** | **82.50** |

## Pair metrics by class

| class | @3/.2 | @4/.2 | @3/.4 | @4/.4 |
|---|---:|---:|---:|---:|
| bed + person | 95 | 99 | 80 | 90 |
| bowl + dining table | 98 | 99 | 97 | 98 |
| horse + person | 78 | 85 | 50 | 64 |
| motorcycle + car | 68 | 74 | 51 | 53 |
| pizza + dining table | 100 | 100 | 100 | 100 |
| sandwich + dining table | 88 | 93 | 84 | 90 |
| truck + car | 88 | 95 | 68 | 73 |
| umbrella + person | 92 | 97 | 83 | 92 |

Raw outputs are stored under
`checkpoints/grpo8_factorized_joint_seed8_20260716/bbox_eval_forced_top3_top4/`.


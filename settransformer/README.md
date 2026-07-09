# Discriminative Set Transformer Probe

This folder contains a bbox-free Set Transformer probe for frozen DINOSAUR slots.
It follows `doc/build_settrabsformer.md`: bottleneck projection, masked SAB layers,
PMA pooling, and training losses that emphasize early-slot marginal separation.
The original random subset-vs-complement loss is disabled by default because it
creates symmetric, contradictory constraints for uniformly sampled subsets.

Default data and checkpoint are the paths from the request.

## Train

```bash
.venv/bin/python -m settransformer.train \
  --output_dir checkpoints/settransformer/top2_clean_anchor009_evidence005
```

The best checkpoint is selected by `valid_selection_score`, not by classification
accuracy. That score rewards single-slot margin range and positive 1-to-2 subset
gain, with a penalty for permutation inconsistency.

## Diagnose

```bash
.venv/bin/python -m settransformer.diagnose \
  --probe_checkpoint checkpoints/settransformer/top2_clean_anchor009_evidence005/settransformer_discriminative_best.pt \
  --out_dir analysis/settransformer/top2_clean_anchor009_evidence005_valid
```

Important outputs:

- `history_metrics.csv`: training losses and validation margin diagnostics.
- `image_slot_ranking.csv`: per-image top slots by true-class single-slot margin
  and first-step marginal gain.
- `summary.json`: subset-size margin curve, first-step gain range, and
  permutation consistency error.

## Design Choices

- Unselected slots are excluded with attention masks, not zero vectors.
- Empty subsets are supported through a learned empty token so selector rewards
  can score the pre-selection state.
- `L_marginal` dynamically finds best/worst added slots for each image and
  trains a positive best gain plus near-zero worst gain.
- `L_consistency` permutes slot order and requires the same subset margin,
  matching the reward probe's intended set semantics.
- `L_comp` is available through `--gamma_comp`, but defaults to `0.0`.

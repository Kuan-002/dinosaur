# GRPO6 Factorized

Clean undirected version of `GRPO8-factorized` for the six-class compositional pair dataset.

The intended difference from GRPO8 is only the rule direction:

- GRPO8 uses directed `anchor -> evidence`.
- GRPO6 uses symmetric `object_a + object_b`.

The training structure is otherwise aligned:

- joint auxiliary + GRPO training from epoch 1
- no warmup and no classifier freeze
- slot component head with image-level MIL supervision
- pair CE over factorized object-pair scores
- GRPO reward from true-class margin increment plus raw component-pair increment
- `hidden_dim=256` and `policy_dim=256`
- checkpoint selection by `valid_acc@3`; pair metrics are logged but not used for selecting the best checkpoint
- default pair completion threshold `0.5`

Default training:

```bash
cd /vol/biomedic3/kw1025/dinosaur
sbatch GRPO6-factorized/slurm_train_grpo6_factorized.sh
```

Default bbox evaluation after training:

```bash
RUN_DIR=GRPO6-factorized/checkpoints/<run_name> \
sbatch GRPO6-factorized/slurm_eval_grpo6_factorized_bbox.sh
```

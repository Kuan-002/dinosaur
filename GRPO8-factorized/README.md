# GRPO8 factorized

This experiment trains on the 8-class COCO rule graph in
`dataset/coco_rule_graph8_v2_area015_012_300_100_100`.

There is no warmup phase. Every minibatch performs two coordinated updates
from epoch 1:

1. an auxiliary update trains the set classifier and the factorized A/E/P
   component heads with random partial slot sets, positive/unknown MIL, and
   pair cross-entropy;
2. a GRPO update trains only the slot-ranking policy using class-margin gain,
   anchor gain, evidence-upweighted gain, and distinct-slot pair gain.

DINOSAUR is frozen and bbox/mask annotations are used only by the final
evaluation.

```bash
sbatch GRPO8-factorized/slurm_train_grpo8_factorized.sh
RUN_DIR=GRPO8-factorized/checkpoints/<run> sbatch GRPO8-factorized/slurm_eval_grpo8_factorized_bbox.sh
```

The matched SetTransformer baseline uses the same raw DINOSAUR slots and data
split. Its bbox ranking is label-free: it predicts the class from the full set,
then greedily adds the slot that most increases that predicted class probability.

```bash
sbatch GRPO8-factorized/slurm_train_settransformer_baseline.sh
```

# SET80 Recon-Only Baseline

This is the original `SET80/train_slothead80.py` experiment with all auxiliary
loss terms disabled and only reconstruction loss active.

Run:

```bash
sbatch SET80/slurm_train_slothead80_recon_only.sh
```

Effective loss weights:

```text
lambda_obj  = 0.0
lambda_geo  = 0.0
lambda_cat  = 0.0
lambda_orth = 0.0
lambda_rec  = 1.0
```

The script intentionally keeps the original SET80 data pipeline and checkpoint
format, but the optimization objective is pure reconstruction:

```text
loss = MSE(decoder([u_obj, u_geo, u_res]), raw_slot)
```


from typing import TypedDict, Optional, Tuple, Dict, List

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SlotAutoencoder
from datasets import get_coco, get_coco_rules, get_pascalVOC
from misc_utils import seed_all, EMA
from ocl_metrics import UnsupervisedMaskIoUMetric


class NoOpLogger:
    def init(self, **kwargs):
        print("wandb disabled")

    def log(self, metrics):
        return None


wandb = NoOpLogger()


def preprocess_batch(
    batch: torch.Tensor | List[torch.Tensor], device: torch.device
):
    if isinstance(batch, list):
        return [img.to(device, non_blocking=device.type == "cuda") for img in batch]
    return [batch.to(device, non_blocking=device.type == "cuda")]


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    mininterval = float(os.environ.get("TQDM_MININTERVAL", "1.0"))
    tqdm_dataloader = tqdm(dataloader, total=len(dataloader), mininterval=mininterval)
    total_loss, n = 0, 0

    mBO = {}
    for j in ["c", "i"]:
        for k in ["", "_slots"]:
            mBO[j + k] = UnsupervisedMaskIoUMetric(
                matching="best_overlap", ignore_background=True, ignore_overlaps=True
            ).to(device)

    for batch in tqdm_dataloader:
        batch = preprocess_batch(batch, device)
        images = batch[0]
        bs = images.shape[0]

        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            out = model.forward(images)
            loss = out["loss"]

        if training:
            loss.backward()
            metrics = {}
            metrics["grad_norm"] = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                metrics["lr"] = scheduler.get_last_lr()[0]
            wandb.log(metrics)
            if ema is not None:
                ema.update()
        else:
            res_init = int(np.sqrt(out["decoder_attn"].shape[-1]))
            decoder_attn = nn.functional.interpolate(
                out["decoder_attn"].view(bs, -1, res_init, res_init),
                size=images.shape[-1],
                mode="bilinear",
            )
            # (B, 1, H, W)
            pred_mask = decoder_attn.argmax(dim=1)
            pred_mask = nn.functional.one_hot(pred_mask).float().permute(0, 3, 1, 2)
            mask_i = nn.functional.one_hot(batch[1]).float().permute(0, 3, 1, 2)
            mask_c = nn.functional.one_hot(batch[2]).float().permute(0, 3, 1, 2)
            mBO["i"].update(pred_mask, mask_i)
            mBO["c"].update(pred_mask, mask_c)

            slot_attn = nn.functional.interpolate(
                out["slot_attn"].view(bs, -1, res_init, res_init),
                size=images.shape[-1],
                mode="bilinear",
            )
            # (B, 1, H, W)
            pred_mask = slot_attn.argmax(dim=1)
            pred_mask = nn.functional.one_hot(pred_mask).float().permute(0, 3, 1, 2)
            mBO["i_slots"].update(pred_mask, mask_i)
            mBO["c_slots"].update(pred_mask, mask_c)

            mBO_i = 100 * mBO["i"].compute()
            mBO_c = 100 * mBO["c"].compute()
            mBO_i_slots = 100 * mBO["i_slots"].compute()
            mBO_c_slots = 100 * mBO["c_slots"].compute()

        n += bs
        total_loss += loss.detach() * bs

        tqdm_dataloader.set_description(
            f"{'train' if training else 'valid'} loss: {total_loss / n:.4f}"
            + (f", lr: {metrics['lr']:.3e}" if training else "")
            + (f", gnorm: {metrics['grad_norm']:,.3f}" if training else "")
            + (
                f", mBO_i: {mBO_i:.2f}, mBO_c: {mBO_c:.2f}"
                + f", mBO_i_slots: {mBO_i_slots:.2f}, mBO_c_slots: {mBO_c_slots:.2f}"
                if not training
                else ""
            ),
            refresh=False,
        )
    if training:
        return dict(loss=total_loss / n)
    return dict(
        loss=total_loss / n,
        mBO_i=mBO_i,
        mBO_c=mBO_c,
        mBO_i_slots=mBO_i_slots,
        mBO_c_slots=mBO_c_slots,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataset", type=str, default="pascal")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "VOC2012_train_val",
            "VOC2012_train_val",
        ),
    )
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument(
        "--subset_csv",
        type=str,
        default="",
        help="Balanced subset CSV for --dataset coco_rules.",
    )
    _nw_help = (
        "DataLoader workers. On Windows, multi-worker loading often fails with shared "
        "memory errors (e.g. 1455); default is 0 there, 8 on Linux/macOS."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0 if sys.platform == "win32" else 8,
        help=_nw_help,
    )
    parser.add_argument("--input_ch", type=int, default=3)
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--num_patches", type=int, default=196)
    parser.add_argument("--embed_dim", type=int, default=768)
    # model
    parser.add_argument(
        "--model_class",
        type=str,
        default="mlp",
        choices=["mlp", "transformer"],
    )
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_slots", type=int, default=6)
    parser.add_argument("--slot_dim", type=int, default=256)
    parser.add_argument("--num_slot_heads", type=int, default=1)
    parser.add_argument("--routing_iters", type=int, default=3)
    parser.add_argument(
        "--sa_topk_patches",
        type=int,
        default=0,
        help="If >0, each slot only aggregates its top-k patch tokens in Slot Attention.",
    )
    parser.add_argument("--proj_cov", action="store_true", default=False)
    # training
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_warmup", type=int, default=10_000)
    parser.add_argument("--lr_half_life", type=int, default=100_000)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--ema_rate", type=float, default=0.999)
    parser.add_argument("--eval_freq", type=int, default=4)
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="mBO_i_slots",
        help="Validation metric used for model selection and early stopping.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop after this many validation runs without monitor improvement. 0 disables it.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum monitor improvement required to reset early-stop patience.",
    )
    parser.add_argument(
        "--collapse_drop_fraction",
        type=float,
        default=0.0,
        help="Stop if monitor drops this fraction below its best value. 0 disables it.",
    )
    parser.add_argument(
        "--early_stop_min_evals",
        type=int,
        default=3,
        help="Minimum validation runs before collapse or patience stopping can trigger.",
    )
    parser.add_argument("--determ", action="store_true", default=False)
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", ""),
        help="W&B entity (username or team). Can also set env WANDB_ENTITY.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="W&B mode. offline/disabled avoids needing a default team on wandb.ai.",
    )
    args = parser.parse_known_args()[0]

    if args.wandb_mode != "disabled":
        import wandb as wandb_module

        wandb = wandb_module

    seed_all(args.seed, args.determ)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Using device: {device} (torch {torch.__version__})")

    if args.dataset == "pascal":
        datasets = get_pascalVOC(args)
    elif args.dataset == "coco":
        datasets = get_coco(args)
    elif args.dataset == "coco_rules":
        datasets = get_coco_rules(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    pin_memory = device.type == "cuda"
    dataloaders = {
        k: DataLoader(
            datasets[k],
            batch_size=args.bs,
            shuffle=(k == "train"),
            drop_last=(k == "train"),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        for k in ["train", "valid"]
    }

    class Config(TypedDict):
        embed_shape: Tuple[int, int]
        decoder_type: str
        hidden_dim: int
        num_blocks: int
        num_heads: int
        dropout: float
        num_slots: int
        slot_dim: int
        num_slot_heads: int
        routing_iters: int
        sa_topk_patches: int
        probabilistic: bool
        proj_cov: bool

    model_kwargs: Config = {
        "embed_shape": (args.num_patches, args.embed_dim),
        "decoder_type": args.model_class,
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "num_slots": args.num_slots,
        "slot_dim": args.slot_dim,
        "num_slot_heads": args.num_slot_heads,
        "routing_iters": args.routing_iters,
        "sa_topk_patches": args.sa_topk_patches,
        "probabilistic":  False,
        "proj_cov": args.proj_cov,
    }

    model = SlotAutoencoder(**model_kwargs)

    model.to(device)
    print(model)
    ema = EMA(model.parameters(), rate=args.ema_rate)

    for k, v in vars(args).items():
        print(f"--{k}={v}")
    print(f"#params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        fused=use_fused,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            step / max(1, args.lr_warmup)
            if step < args.lr_warmup
            else (0.5 ** (1 / args.lr_half_life)) ** (step - args.lr_warmup)
        ),
    )

    wandb_kwargs: Dict[str, object] = {
        "project": "psa",
        "name": args.exp_name,
        "config": vars(args),
    }
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
    wandb_mode = args.wandb_mode
    if wandb_mode == "online" and not args.wandb_entity:
        print(
            "wandb: no --wandb_entity / WANDB_ENTITY; using offline "
            "(local ./wandb). Pass --wandb_entity <team_or_user> for cloud."
        )
        wandb_mode = "offline"
    wandb_kwargs["mode"] = wandb_mode
    wandb.init(**wandb_kwargs)
    best_loss = 1e6
    best_mbo_i_slots = -1.0
    best_mbo_c_slots = -1.0
    best_monitor = -float("inf")
    stale_evals = 0
    eval_count = 0
    start_t = time.time()

    def save_checkpoint(name: str, step: int, metrics: Dict[str, float]) -> str:
        save_path = f"./checkpoints/{args.exp_name}/{name}"
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(
            {
                "args": vars(args),
                "step": step,
                "metrics": metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path,
        )
        return save_path

    for i in range(args.epochs):
        train_metrics = run_epoch(
            model, dataloaders["train"], device, ema, optimizer, scheduler
        )
        wandb.log({"train_" + k: v for k, v in train_metrics.items()})

        if (i % args.eval_freq) == 0:
            model.eval()
            ema.apply()  # apply ema model weights
            valid_metrics = run_epoch(model, dataloaders["valid"], device)
            wandb.log({"valid_" + k: v for k, v in valid_metrics.items()})

            step = (i + 1) * len(dataloaders["train"])
            t = time.strftime("[%H:%M:%S, %d/%m/%Y]")
            elapsed_t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_t))
            print(
                f"{t} epoch: {i + 1}, step: {step}, elapsed: {elapsed_t}"
                f"\n{t} train {', '.join(f'{k}: {v:.5f}' for k, v in train_metrics.items())}"
                f"\n{t} valid {', '.join(f'{k}: {v:.5f}' for k, v in valid_metrics.items())}"
            )

            if args.monitor_metric not in valid_metrics:
                raise ValueError(
                    f"--monitor_metric={args.monitor_metric!r} is not in validation "
                    f"metrics: {sorted(valid_metrics)}"
                )

            eval_count += 1
            monitor_value = valid_metrics[args.monitor_metric]
            improved_monitor = (
                monitor_value > best_monitor + args.early_stop_min_delta
            )
            if improved_monitor:
                best_monitor = monitor_value
                stale_evals = 0
            else:
                stale_evals += 1

            if valid_metrics["loss"] < best_loss:
                best_loss = valid_metrics["loss"]
                save_path = save_checkpoint("checkpoint.pt", step, valid_metrics)
                print(f"{t} best loss model saved: {save_path}")
            if valid_metrics["mBO_i_slots"] > best_mbo_i_slots:
                best_mbo_i_slots = valid_metrics["mBO_i_slots"]
                save_path = save_checkpoint(
                    "checkpoint_best_mbo_i_slots.pt", step, valid_metrics
                )
                print(f"{t} best mBO_i_slots model saved: {save_path}")
            if valid_metrics["mBO_c_slots"] > best_mbo_c_slots:
                best_mbo_c_slots = valid_metrics["mBO_c_slots"]
                save_path = save_checkpoint(
                    "checkpoint_best_mbo_c_slots.pt", step, valid_metrics
                )
                print(f"{t} best mBO_c_slots model saved: {save_path}")

            should_stop = False
            stop_reason = ""
            can_stop = eval_count >= args.early_stop_min_evals
            if (
                can_stop
                and args.collapse_drop_fraction > 0
                and best_monitor > 0
                and monitor_value < best_monitor * (1.0 - args.collapse_drop_fraction)
            ):
                should_stop = True
                stop_reason = (
                    f"{args.monitor_metric} collapsed from best {best_monitor:.5f} "
                    f"to {monitor_value:.5f}"
                )
            elif (
                can_stop
                and args.early_stop_patience > 0
                and stale_evals >= args.early_stop_patience
            ):
                should_stop = True
                stop_reason = (
                    f"{args.monitor_metric} did not improve for "
                    f"{stale_evals} validation runs"
                )

            ema.restore()  # restore model weights
            if should_stop:
                print(f"{t} early stop: {stop_reason}")
                break

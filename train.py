from typing import TypedDict, Optional, Tuple, Dict, List

import os
import sys
import time
import wandb
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SlotAutoencoder
from datasets import get_pascalVOC
from misc_utils import seed_all, EMA
from ocl_metrics import UnsupervisedMaskIoUMetric


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
    parser.add_argument("--model_class", type=str, default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_slots", type=int, default=6)
    parser.add_argument("--slot_dim", type=int, default=256)
    parser.add_argument("--num_slot_heads", type=int, default=1)
    parser.add_argument("--routing_iters", type=int, default=3)
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
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode. offline/disabled avoids needing a default team on wandb.ai.",
    )
    args = parser.parse_known_args()[0]

    seed_all(args.seed, args.determ)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Using device: {device} (torch {torch.__version__})")

    datasets = get_pascalVOC(args)
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
        num_slots: int
        slot_dim: int
        num_slot_heads: int
        routing_iters: int
        probabilistic: bool
        proj_cov: bool

    model_kwargs: Config = {
        "embed_shape": (args.num_patches, args.embed_dim),
        "num_slots": args.num_slots,
        "slot_dim": args.slot_dim,
        "num_slot_heads": args.num_slot_heads,
        "routing_iters": args.routing_iters,
        "probabilistic":  False,
        "proj_cov": args.proj_cov,
    }

    if args.model_class == "mlp":
        model = SlotAutoencoder(hidden_dim=args.hidden_dim, **model_kwargs)
    else:
        raise NotImplementedError

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
    start_t = time.time()

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

            if valid_metrics["loss"] < best_loss:
                best_loss = valid_metrics["loss"]
                save_path = f"./checkpoints/{args.exp_name}/checkpoint.pt"
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save(
                    {
                        "args": vars(args),
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    save_path,
                )
                print(f"{t} model saved: {save_path}")
            ema.restore()  # restore model weights

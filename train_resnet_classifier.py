import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "TORCH_HOME",
    str(Path(__file__).resolve().parent / ".cache" / "torch"),
)

import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from misc_utils import seed_all
from train_slot_classifier import build_dataset, build_transforms, make_loader, subset_dataset


class EpochMetrics:
    def __init__(self, loss: float, acc: float):
        self.loss = loss
        self.acc = acc


def resolve_resnet(name: str, weights_name: str):
    name = name.lower()
    weights_name = weights_name.lower()
    if name == "resnet18":
        weights_cls = models.ResNet18_Weights
        ctor = models.resnet18
    elif name == "resnet50":
        weights_cls = models.ResNet50_Weights
        ctor = models.resnet50
    else:
        raise ValueError(f"Unsupported --arch={name!r}; use resnet18 or resnet50")

    if weights_name == "none":
        weights = None
    elif weights_name in ("imagenet", "default"):
        weights = weights_cls.DEFAULT
    else:
        weights = weights_cls[weights_name]
    return ctor, weights


class FrozenResNetMLP(nn.Module):
    def __init__(self, arch: str, weights_name: str, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        ctor, weights = resolve_resnet(arch, weights_name)
        self.backbone = ctor(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.backbone.requires_grad_(False)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(images)
        return self.head(features)


class FinetuneResNet(nn.Module):
    def __init__(self, arch: str, weights_name: str, num_classes: int, dropout: float):
        super().__init__()
        ctor, weights = resolve_resnet(arch, weights_name)
        self.backbone = ctor(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.head = self.backbone.fc

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


def run_epoch(model, dataloader, device, criterion, optimizer=None) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    if hasattr(model, "backbone") and isinstance(model, FrozenResNetMLP):
        model.backbone.eval()
        model.head.train(training)

    total_loss = 0.0
    total_correct = 0
    total = 0
    desc = "train" if training else "valid"
    for images, targets in tqdm(dataloader, desc=desc, mininterval=1.0):
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                1.0,
            )
            optimizer.step()
        bs = targets.numel()
        total_loss += float(loss.detach()) * bs
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total += bs
    return EpochMetrics(total_loss / total, total_correct / total)


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/vol/biomedic3/kw1025/dinosaur/dataset/classification_dataset_clean_600_200_200")
    parser.add_argument("--output_dir", default="./checkpoints/resnet_classifier_clean_600_200_200")
    parser.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--weights", default="imagenet", help="imagenet/default, none, or torchvision enum key")
    parser.add_argument("--mode", default="freeze", choices=["freeze", "finetune"])
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    parser.add_argument("--eval_test", action="store_true", default=False)
    parser.add_argument("--eval_confounding", action="store_true", default=False)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--early_stop_min_epochs", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001)
    args = parser.parse_args()

    seed_all(args.seed, False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (torch {torch.__version__})")

    tfm = build_transforms(args.input_res)
    train_set = subset_dataset(
        build_dataset(args.data, "train", tfm["train"]),
        args.quick_limit_train,
        args.seed,
    )
    valid_set = subset_dataset(
        build_dataset(args.data, "valid", tfm["valid"]),
        args.quick_limit_val,
        args.seed,
    )
    classes = getattr(train_set, "classes", getattr(getattr(train_set, "dataset", None), "classes", None))
    num_classes = len(classes) if classes is not None else 10
    print(f"train={len(train_set)} valid={len(valid_set)} classes={num_classes}")

    train_loader = make_loader(train_set, args, device, shuffle=True)
    valid_loader = make_loader(valid_set, args, device)

    if args.mode == "freeze":
        model = FrozenResNetMLP(args.arch, args.weights, args.hidden_dim, num_classes, args.dropout)
        optimizer_lr = args.lr
    else:
        model = FinetuneResNet(args.arch, args.weights, num_classes, args.dropout)
        optimizer_lr = args.finetune_lr
    model.to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"resnet benchmark: arch={args.arch}, weights={args.weights}, mode={args.mode}, "
        f"dropout={args.dropout}, trainable_params={n_trainable:,}, total_params={n_total:,}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=optimizer_lr, weight_decay=args.wd)
    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
        valid_metrics = run_epoch(model, valid_loader, device, criterion)
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        row = {
            "epoch": epoch,
            "elapsed": elapsed,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.acc,
            "valid_loss": valid_metrics.loss,
            "valid_acc": valid_metrics.acc,
        }
        history.append(row)
        print(
            f"epoch={epoch} elapsed={elapsed} "
            f"train_loss={train_metrics.loss:.4f} train_acc={100 * train_metrics.acc:.2f} "
            f"valid_loss={valid_metrics.loss:.4f} valid_acc={100 * valid_metrics.acc:.2f}"
        )

        improved = valid_metrics.acc > best_acc + args.early_stop_min_delta
        if improved:
            best_acc = valid_metrics.acc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "args": vars(args),
                    "classes": classes,
                    "epoch": epoch,
                    "valid_acc": valid_metrics.acc,
                    "model_state_dict": model.state_dict(),
                },
                os.path.join(args.output_dir, "checkpoint_best.pt"),
            )
            print(f"saved best classifier: {os.path.join(args.output_dir, 'checkpoint_best.pt')}")
        else:
            stale_epochs += 1

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

        if (
            args.early_stop_patience > 0
            and epoch >= args.early_stop_min_epochs
            and stale_epochs >= args.early_stop_patience
        ):
            print(
                "early_stop: "
                f"best_epoch={best_epoch}, best_valid_acc={100 * best_acc:.2f}, "
                f"stale_epochs={stale_epochs}"
            )
            break

    print(f"best_valid_acc={100 * best_acc:.2f} best_epoch={best_epoch}")

    best_path = os.path.join(args.output_dir, "checkpoint_best.pt")
    final_metrics = {"best_valid_acc": best_acc, "best_epoch": best_epoch}
    if os.path.exists(best_path) and (args.eval_test or args.eval_confounding):
        best_ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
        model.to(device)
        model.eval()

    if args.eval_test:
        test_set = build_dataset(args.data, "test", tfm["valid"])
        test_loader = make_loader(test_set, args, device)
        test_metrics = run_epoch(model, test_loader, device, criterion)
        final_metrics.update(test_loss=test_metrics.loss, test_acc=test_metrics.acc)
        print(f"test loss={test_metrics.loss:.4f} test_acc={100 * test_metrics.acc:.2f}")

    if args.eval_confounding:
        confounding_set = build_dataset(args.data, "confounding_test", tfm["valid"])
        confounding_loader = make_loader(confounding_set, args, device)
        confounding_metrics = run_epoch(model, confounding_loader, device, criterion)
        final_metrics.update(
            confounding_loss=confounding_metrics.loss,
            confounding_acc=confounding_metrics.acc,
        )
        print(
            f"confounding loss={confounding_metrics.loss:.4f} "
            f"confounding_acc={100 * confounding_metrics.acc:.2f}"
        )

    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)


if __name__ == "__main__":
    main()

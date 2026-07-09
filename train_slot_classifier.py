import argparse
import json
import os
import random
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

os.environ.setdefault(
    "TORCH_HOME",
    str(Path(__file__).resolve().parent / ".cache" / "torch"),
)

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models import SlotAutoencoder
from misc_utils import seed_all


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ZipImageFolder(Dataset):
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, zip_path: str, split: str, transform=None):
        self.zip_path = zip_path
        self.split = "val" if split == "valid" else split
        self.transform = transform
        self._zip: Optional[zipfile.ZipFile] = None

        with zipfile.ZipFile(self.zip_path) as zf:
            names = zf.namelist()
            info_name = next(
                (n for n in names if n.endswith("dataset_info.json")),
                None,
            )
            if info_name is not None:
                with zf.open(info_name) as f:
                    info = json.load(f)
                self.classes = list(info["classes"])
                self.class_to_idx = dict(info["class_to_idx"])
            else:
                classes = sorted(
                    {
                        parts[-2]
                        for n in names
                        if f"/{self.split}/" in n
                        for parts in [n.split("/")]
                        if len(parts) >= 3 and n.lower().endswith(self.IMG_EXTENSIONS)
                    }
                )
                self.classes = classes
                self.class_to_idx = {name: idx for idx, name in enumerate(classes)}

            self.samples = []
            split_marker = f"/{self.split}/"
            for name in names:
                if not name.lower().endswith(self.IMG_EXTENSIONS):
                    continue
                if split_marker not in name:
                    continue
                parts = name.split("/")
                class_name = parts[-2]
                if class_name in self.class_to_idx:
                    self.samples.append((name, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(f"No images found for split={self.split!r} in {zip_path}")

    def __len__(self):
        return len(self.samples)

    def _get_zip(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        return self._zip

    def __getitem__(self, idx):
        name, target = self.samples[idx]
        with self._get_zip().open(name) as f:
            img = Image.open(BytesIO(f.read())).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class SlotMLPClassifier(nn.Module):
    def __init__(self, backbone: SlotAutoencoder, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        flat_dim = backbone.num_slots * backbone.slot_dim
        self.head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_slots(self, images: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone.forward_dino(images)
            features = self.backbone.mlp(features)
            slots, _, _ = self.backbone.slot_attention(features)
        return slots.flatten(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_slots(images))


@dataclass
class EpochMetrics:
    loss: float
    acc: float


def build_transforms(input_res: int):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return {
        "train": transforms.Compose(
            [
                transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(input_res),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(input_res),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    }


def build_dataset(data: str, split: str, transform):
    path = Path(data)
    folder_split = "val" if split == "valid" else split
    if path.is_file() and path.suffix == ".zip":
        return ZipImageFolder(str(path), split=split, transform=transform)
    root = path / "classification_dataset"
    if root.is_dir():
        path = root
    return datasets.ImageFolder(str(path / folder_split), transform=transform)


def subset_dataset(dataset: Dataset, limit: int, seed: int):
    if limit <= 0 or limit >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:limit])


def load_backbone(checkpoint_path: str, device: torch.device) -> SlotAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    model = SlotAutoencoder(
        embed_shape=(
            int(ckpt_args.get("num_patches", 196)),
            int(ckpt_args.get("embed_dim", 768)),
        ),
        decoder_type=ckpt_args.get("model_class", "mlp"),
        hidden_dim=int(ckpt_args.get("hidden_dim", 2048)),
        num_blocks=int(ckpt_args.get("num_blocks", 6)),
        num_heads=int(ckpt_args.get("num_heads", 6)),
        dropout=float(ckpt_args.get("dropout", 0.0)),
        num_slots=int(ckpt_args.get("num_slots", 8)),
        slot_dim=int(ckpt_args.get("slot_dim", 256)),
        num_slot_heads=int(ckpt_args.get("num_slot_heads", 1)),
        routing_iters=int(ckpt_args.get("routing_iters", 3)),
        sa_topk_patches=int(ckpt_args.get("sa_topk_patches", 0)),
        probabilistic=False,
        proj_cov=bool(ckpt_args.get("proj_cov", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    return model


def run_epoch(model, dataloader, device, criterion, optimizer=None) -> EpochMetrics:
    training = optimizer is not None
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
            nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            optimizer.step()
        batch_size = targets.numel()
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total += batch_size
    return EpochMetrics(loss=total_loss / total, acc=total_correct / total)


def make_loader(dataset, args, device, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/vol/biomedic3/kw1025/dinosaur/dataset/coco_scene_guidelines_10_v2_classification.zip")
    parser.add_argument("--checkpoint", default="/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt")
    parser.add_argument("--output_dir", default="./checkpoints/slot_classifier_coco_scene_guidelines_10_v2")
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    parser.add_argument("--eval_test", action="store_true", default=False)
    parser.add_argument("--eval_confounding", action="store_true", default=False)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_epochs", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    args = parser.parse_args()

    seed_all(args.seed, False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
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

    backbone = load_backbone(args.checkpoint, device)
    model = SlotMLPClassifier(backbone, args.hidden_dim, num_classes, args.dropout).to(device)
    print(
        "backbone slots: "
        f"num_slots={backbone.num_slots}, slot_dim={backbone.slot_dim}, "
        f"classifier_input_dim={backbone.num_slots * backbone.slot_dim}"
    )
    print(f"trainable params: {sum(p.numel() for p in model.head.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.wd)
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
                    "head_state_dict": model.head.state_dict(),
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
    final_metrics = {
        "best_valid_acc": best_acc,
        "best_epoch": best_epoch,
    }
    if os.path.exists(best_path) and (args.eval_test or args.eval_confounding):
        best_ckpt = torch.load(best_path, map_location="cpu")
        model.head.load_state_dict(best_ckpt["head_state_dict"], strict=True)
        model.eval()

    if args.eval_test:
        test_set = build_dataset(args.data, "test", tfm["valid"])
        test_loader = make_loader(test_set, args, device)
        test_metrics = run_epoch(model, test_loader, device, criterion)
        final_metrics.update(
            test_loss=test_metrics.loss,
            test_acc=test_metrics.acc,
        )
        print(
            f"test loss={test_metrics.loss:.4f} "
            f"test_acc={100 * test_metrics.acc:.2f}"
        )

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

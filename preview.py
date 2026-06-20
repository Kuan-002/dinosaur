import argparse
import os
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from datasets import get_pascalVOC
from models import SlotAutoencoder


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm_to_uint8(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def make_palette(n: int = 256) -> np.ndarray:
    rng = np.random.default_rng(0)
    palette = np.zeros((n, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    palette[1:] = rng.integers(0, 255, size=(n - 1, 3), dtype=np.uint8)
    return palette


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    idx = np.clip(mask, 0, len(palette) - 1)
    return palette[idx]


def overlay(rgb: np.ndarray, seg_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = (1 - alpha) * rgb.astype(np.float32) + alpha * seg_rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args: Dict = ckpt["args"]
    model = SlotAutoencoder(
        embed_shape=(ckpt_args["num_patches"], ckpt_args["embed_dim"]),
        hidden_dim=ckpt_args["hidden_dim"],
        num_slots=ckpt_args["num_slots"],
        slot_dim=ckpt_args["slot_dim"],
        num_slot_heads=ckpt_args["num_slot_heads"],
        routing_iters=ckpt_args["routing_iters"],
        probabilistic=False,
        proj_cov=ckpt_args["proj_cov"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/voc2012_local/checkpoint.pt",
    )
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid"])
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="./previews")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model, ckpt_args = load_model(args.checkpoint, device)
    data_dir = args.data_dir if args.data_dir else ckpt_args["data_dir"]
    ds_args = SimpleNamespace(
        data_dir=data_dir,
        input_res=ckpt_args["input_res"],
        cache=False,
    )
    dataset = get_pascalVOC(ds_args)[args.split]
    sample = dataset[args.index]
    image = sample[0] if isinstance(sample, tuple) else sample
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(image)

    bs, _, h, w = image.shape
    res_init = int(np.sqrt(out["decoder_attn"].shape[-1]))
    decoder_attn = F.interpolate(
        out["decoder_attn"].view(bs, -1, res_init, res_init),
        size=h,
        mode="bilinear",
    )
    slot_attn = F.interpolate(
        out["slot_attn"].view(bs, -1, res_init, res_init),
        size=h,
        mode="bilinear",
    )

    dec_mask = decoder_attn.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    slot_mask = slot_attn.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    input_rgb = denorm_to_uint8(image[0])
    palette = make_palette()
    dec_rgb = colorize_mask(dec_mask, palette)
    slot_rgb = colorize_mask(slot_mask, palette)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.split}_{args.index:05d}"
    Image.fromarray(input_rgb).save(os.path.join(args.out_dir, f"{stem}_input.png"))
    Image.fromarray(dec_rgb).save(os.path.join(args.out_dir, f"{stem}_decoder_mask.png"))
    Image.fromarray(slot_rgb).save(os.path.join(args.out_dir, f"{stem}_slot_mask.png"))
    Image.fromarray(overlay(input_rgb, dec_rgb)).save(
        os.path.join(args.out_dir, f"{stem}_decoder_overlay.png")
    )
    Image.fromarray(overlay(input_rgb, slot_rgb)).save(
        os.path.join(args.out_dir, f"{stem}_slot_overlay.png")
    )

    print(f"Saved preview images to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

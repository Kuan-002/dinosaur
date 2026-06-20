from typing import Callable, Optional, List, Dict

import os
import torch
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

ImageFile.LOAD_TRUNCATED_IMAGES = True


def cache_data(load_fn: Callable[[str], np.ndarray], files: List[str], name: str):
    with ThreadPoolExecutor() as executor:
        return list(
            tqdm(
                executor.map(load_fn, files),
                total=len(files),
                desc=f"Caching {name}",
                mininterval=15,
            )
        )


class PascalVOC(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable,
        mask_transform: Optional[Callable] = None,
        cache: bool = False,
    ):
        assert split in ["train", "valid"]
        self.root = root
        if split == "train":
            seg_lists = os.path.join(self.root, "ImageSets/Segmentation")
            if os.path.isfile(os.path.join(seg_lists, "trainaug.txt")):
                self.split = "trainaug"
            elif os.path.isfile(os.path.join(seg_lists, "train.txt")):
                self.split = "train"
            else:
                raise FileNotFoundError(
                    f"Expected trainaug.txt or train.txt in {seg_lists}"
                )
        else:
            self.split = split[:3]  # "valid" -> "val"
        self.transform = transform
        self.mask_transform = mask_transform
        self.cache = cache

        self.images = []
        with open(
            os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt"), "r"
        ) as file:
            for line in file:
                self.images.append(line.strip())

        if self.cache:
            self.cached_data = {}
            folder_names = ["JPEGImages"]
            if self.split == "val":
                folder_names += ["SegmentationClass", "SegmentationObject"]
            for k in folder_names:
                ext = ".jpg" if k == "JPEGImages" else ".png"
                self.cached_data[k] = cache_data(
                    load_fn=lambda x: self._load_image(x),
                    files=[
                        os.path.join(self.root, f"{k}", i) + ext for i in self.images
                    ],
                    name=f"PascalVOC {self.split} {k}",
                )

    def __len__(self):
        return len(self.images)

    def _load_image(self, file: str) -> np.ndarray:
        with Image.open(file) as img:
            img = np.array(img, dtype=np.uint8)
        return img

    def __getitem__(self, idx: int):

        if self.cache:
            img = transforms.ToPILImage()(self.cached_data["JPEGImages"][idx])
        else:
            name = self.images[idx]
            img = Image.open(os.path.join(self.root, "JPEGImages", name + ".jpg"))

        img = self.transform(img)

        if self.split in ("trainaug", "train"):
            return img
        elif self.split == "val":

            if self.cache:
                mask_class = transforms.ToPILImage()(
                    self.cached_data["SegmentationClass"][idx]
                )
                mask_instance = transforms.ToPILImage()(
                    self.cached_data["SegmentationObject"][idx]
                )
            else:
                mask_class = Image.open(
                    os.path.join(self.root, "SegmentationClass", name) + ".png"
                )
                mask_instance = Image.open(
                    os.path.join(self.root, "SegmentationObject", name) + ".png"
                )

            assert self.mask_transform
            mask_class = self.mask_transform(mask_class).squeeze().long()
            mask_class[mask_class == 255] = 0  # Ignore objects' boundaries

            mask_instance = self.mask_transform(mask_instance).squeeze().long()
            mask_instance[mask_instance == 255] = 0
            ignore_mask = torch.zeros(1, *mask_class.shape).long()  # no overlap
            return img, mask_instance, mask_class, ignore_mask


def get_pascalVOC(args: argparse.Namespace) -> Dict[str, PascalVOC]:

    transform = dict(
        train=transforms.Compose(
            [
                transforms.Resize(
                    size=args.input_res,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomCrop(args.input_res),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        eval=transforms.Compose(
            [
                transforms.Resize(
                    size=args.input_res,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(size=args.input_res),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )
    mask_res = args.input_res
    mask_transform = transforms.Compose(
        [
            transforms.Resize(
                size=mask_res,
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.CenterCrop(size=mask_res),
            transforms.PILToTensor(),
        ]
    )

    datasets = {
        k: PascalVOC(
            root=args.data_dir,
            split=k,
            transform=transform[(k if k == "train" else "eval")],
            mask_transform=(mask_transform if k != "train" else None),
            cache=args.cache,
        )
        for k in ["train", "valid"]
    }
    return datasets

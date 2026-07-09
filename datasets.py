from typing import Callable, Optional, List, Dict

import os
import csv
import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFile
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


class CocoRules(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable,
        subset_csv: Optional[str],
        mask_transform: Optional[Callable] = None,
    ):
        assert split in ["train", "valid"]
        self.root = root
        self.split = "train" if split == "train" else "val"
        self.transform = transform
        self.mask_transform = mask_transform

        if subset_csv:
            if not os.path.isfile(subset_csv):
                raise FileNotFoundError(f"COCO rules subset CSV not found: {subset_csv}")
            with open(subset_csv, newline="") as f:
                rows = [row for row in csv.DictReader(f) if row["split"] == self.split]
            self.samples = rows
            self.image_ids = [int(row["image_id"]) for row in rows]
            self.source_splits = [
                row.get("source_split") or row["split"]
                for row in rows
            ]
        else:
            self.source_splits = [self.split]
            data = self._load_annotation_data(self.split)
            images_by_id = {int(img["id"]): img for img in data["images"]}
            self.image_ids = sorted(images_by_id)
            self.samples = [
                {
                    "split": self.split,
                    "source_split": self.split,
                    "image_id": str(image_id),
                    "file_name": images_by_id[image_id]["file_name"],
                }
                for image_id in self.image_ids
            ]
            self.source_splits = [self.split for _ in self.samples]

        self.source_by_image: dict[int, str] = {}
        for row, source_split in zip(self.samples, self.source_splits):
            self.source_by_image[int(row["image_id"])] = source_split

        self.images_by_key: dict[tuple[str, int], dict] = {}
        self.annotations_by_key: dict[tuple[str, int], list[dict]] = {}
        wanted_by_source: dict[str, set[int]] = {}
        for image_id, source_split in zip(self.image_ids, self.source_splits):
            wanted_by_source.setdefault(source_split, set()).add(image_id)
            self.annotations_by_key[(source_split, image_id)] = []

        for source_split, wanted in wanted_by_source.items():
            data = self._load_annotation_data(source_split)
            for img in data["images"]:
                image_id = int(img["id"])
                if image_id in wanted:
                    self.images_by_key[(source_split, image_id)] = img
            for ann in data["annotations"]:
                image_id = int(ann["image_id"])
                if image_id in wanted and not ann.get("iscrowd", 0):
                    self.annotations_by_key.setdefault((source_split, image_id), []).append(ann)

    def __len__(self):
        return len(self.samples)

    def _load_annotation_data(self, source_split: str) -> dict:
        annotation_file = os.path.join(
            self.root, "annotations", f"instances_{source_split}2017.json"
        )
        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")
        with open(annotation_file) as f:
            return json.load(f)

    def _source_split_for(self, image_id: int) -> str:
        return self.source_by_image.get(image_id, self.split)

    def _load_image(self, image_id: int) -> Image.Image:
        source_split = self._source_split_for(image_id)
        info = self.images_by_key[(source_split, image_id)]
        image_dir = os.path.join(self.root, f"{source_split}2017")
        path = os.path.join(image_dir, info["file_name"])
        return Image.open(path).convert("RGB")

    def _draw_polygon_mask(self, image_id: int) -> tuple[Image.Image, Image.Image]:
        source_split = self._source_split_for(image_id)
        info = self.images_by_key[(source_split, image_id)]
        width, height = int(info["width"]), int(info["height"])
        mask_instance = Image.new("I", (width, height), 0)
        mask_class = Image.new("I", (width, height), 0)
        draw_instance = ImageDraw.Draw(mask_instance)
        draw_class = ImageDraw.Draw(mask_class)

        instance_id = 1
        for ann in self.annotations_by_key.get((source_split, image_id), []):
            segmentation = ann.get("segmentation")
            if not isinstance(segmentation, list):
                continue
            category_id = int(ann["category_id"])
            drew_any = False
            for polygon in segmentation:
                if len(polygon) < 6:
                    continue
                points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
                draw_instance.polygon(points, fill=instance_id)
                draw_class.polygon(points, fill=category_id)
                drew_any = True
            if drew_any:
                instance_id += 1

        return mask_instance, mask_class

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img = self.transform(self._load_image(image_id))

        if self.split == "train":
            return img

        assert self.mask_transform
        mask_instance, mask_class = self._draw_polygon_mask(image_id)
        mask_instance = self.mask_transform(mask_instance).squeeze().long()
        mask_class = self.mask_transform(mask_class).squeeze().long()
        ignore_mask = torch.zeros(1, *mask_class.shape).long()
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


def _get_coco_instances(args: argparse.Namespace, subset_csv: Optional[str]) -> Dict[str, CocoRules]:
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
    mask_transform = transforms.Compose(
        [
            transforms.Resize(
                size=args.input_res,
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.CenterCrop(size=args.input_res),
            transforms.PILToTensor(),
        ]
    )

    return {
        k: CocoRules(
            root=args.data_dir,
            split=k,
            transform=transform[(k if k == "train" else "eval")],
            mask_transform=(mask_transform if k != "train" else None),
            subset_csv=subset_csv,
        )
        for k in ["train", "valid"]
    }


def get_coco_rules(args: argparse.Namespace) -> Dict[str, CocoRules]:
    subset_csv = args.subset_csv
    if not subset_csv:
        subset_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "analysis",
            "coco_and_scene_dataset",
            "coco_scene_guidelines_10_v2",
            "balanced_samples.csv",
        )
    return _get_coco_instances(args, subset_csv=subset_csv)


def get_coco(args: argparse.Namespace) -> Dict[str, CocoRules]:
    return _get_coco_instances(args, subset_csv=None)

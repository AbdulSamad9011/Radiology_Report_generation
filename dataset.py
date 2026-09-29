"""
Dataset loader for chest X-ray report generation.

Expects an annotation JSON in the widely-used preprocessed "IU X-Ray"
format (the same layout used by the R2Gen paper, Chen et al. 2020):

    {
      "train": [
        {"id": "CXR1000_IM-0003", "report": "the heart is normal...",
         "image_path": ["CXR1000_IM-0003-1001.png", "CXR1000_IM-0003-2001.png"]},
        ...
      ],
      "val": [...],
      "test": [...]
    }

Each study typically has a frontal + lateral view (image_path has 2
entries); some have only one. See README.md for where to get this file.
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int = 224, train: bool = True):
    # Note: we deliberately do NOT horizontally flip chest X-rays -
    # left/right laterality is clinically meaningful, unlike in most
    # natural-image augmentation pipelines.
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class IUXrayDataset(Dataset):
    def __init__(self, annotation_path, image_dir, vocab, split="train",
                 image_size=224, max_len=100, max_images=2):
        with open(annotation_path) as f:
            ann = json.load(f)
        if split not in ann:
            raise ValueError(f"Split '{split}' not in annotation file (keys: {list(ann.keys())})")

        self.samples = ann[split]
        self.image_dir = image_dir
        self.vocab = vocab
        self.max_len = max_len
        self.max_images = max_images
        self.transform = build_transform(image_size, train=(split == "train"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_paths = item["image_path"][: self.max_images]

        images = []
        for p in image_paths:
            img = Image.open(os.path.join(self.image_dir, p)).convert("RGB")
            images.append(self.transform(img))

        # Some studies only have one view - pad by repeating the last view
        # so every batch element has a fixed number of views.
        while len(images) < self.max_images:
            images.append(images[-1].clone())
        images = torch.stack(images, dim=0)  # (max_images, 3, H, W)

        report_ids = self.vocab.encode(item["report"], max_len=self.max_len)
        report_tensor = torch.tensor(report_ids, dtype=torch.long)

        return images, report_tensor

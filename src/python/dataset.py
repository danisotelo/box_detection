import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from torchvision.transforms import (
    RandomHorizontalFlip,
    ColorJitter,
    RandomRotation,
    ToTensor,
    Compose,
    RandomChoice,
    Normalize,
)


class BoxDataset(Dataset):
    """
    Custom PyTorch Dataset to load and preprocess box detection data.

    Args:
        data_dir (str): Path to the dataset directory.
        transforms (callable): Transformations to apply to the data.

    Methods:
        __len__(): Returns the size of the dataset.
        __getitem__(idx) : Returns a preprocessed image and its corresponding annotation.
    """

    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.image_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".png")
        ]
        self.annotation_paths = [
            os.path.join(data_dir, f.replace(".png", ".json"))
            for f in os.listdir(data_dir)
            if f.endswith(".png")
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load annotations
        with open(annotation_path, "r") as f:
            annotation = json.load(f)

        # Parse annotations
        boxes = []
        masks = []
        for shape in annotation["shapes"]:
            points = np.array(shape["points"])
            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
            boxes.append([xmin, ymin, xmax, ymax])
            mask = np.zeros((img.height, img.width), dtype=np.uint8)
            points = np.round(points).astype(np.int32)
            mask = cv2.fillPoly(mask, [points], 1)
            masks.append(mask)

        # Convert to tensors
        boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks}

        if self.transforms:
            img = self.transforms(img)

        return img, target


def get_transforms(train=True):
    """
    Defines data augmentation and preprocessing pipeline.

    Args:
        train (bool): If True, applies training augmentations.

    Returns:
        callable: Composed transformations.
    """
    transforms = [ToTensor()]  # Always convert to tensor

    if train:
        moderate_augmentations = [
            RandomHorizontalFlip(p=0.5),  # Horizontal flip
            RandomRotation(degrees=90),  # Rotate within ±10 degrees
            ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            ),  # Adjust colors
        ]

        # Randomly apply one augmentation (or none)
        transforms.append(RandomChoice(moderate_augmentations))

    # Normalize based on the pretrained backbone
    # These are the standard ImageNet normalization values
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return Compose(transforms)

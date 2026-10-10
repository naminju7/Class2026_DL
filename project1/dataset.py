import os
import csv
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_class_names(data_dir):
    """class_mapping.json -> (작물 이름 리스트, 건강 상태 이름 리스트)"""
    with open(os.path.join(data_dir, "class_mapping.json"), encoding="utf-8") as f:
        m = json.load(f)
    crop_names = [m["crop_label"][str(i)] for i in range(len(m["crop_label"]))]
    health_names = [m["health_label"][str(i)] for i in range(len(m["health_label"]))]
    return crop_names, health_names


class PlantDataset(Dataset):
    """split 은 "train" 처럼 문자열 하나이거나 ["train", "valid"] 처럼 리스트(두 데이터를 합칠 때).

    cache=True 이면 이미지를 한 번만 디코딩해서 메모리에 올려 둔다 (8,000장 약 400MB).
    limit 은 스모크 테스트용으로 앞의 N장만 쓸 때 사용한다.
    """

    def __init__(self, data_dir, split, transform=None, cache=True, limit=None):
        self.transform = transform
        splits = [split] if isinstance(split, str) else list(split)
        self.items = []  # (이미지 경로, crop, health)
        for sp in splits:
            img_dir = os.path.join(data_dir, sp, "images")
            # train.csv 앞에 BOM이 있으므로 utf-8-sig 로 읽는다
            with open(os.path.join(data_dir, f"{sp}.csv"), encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    self.items.append((os.path.join(img_dir, row["id"]),
                                       int(row["crop_label"]), int(row["health_label"])))
        if limit is not None:
            self.items = self.items[:limit]
        self.images = None
        if cache:
            self.images = [self._open(p) for p, _, _ in self.items]

    @staticmethod
    def _open(path):
        img = Image.open(path).convert("RGB")
        img.load()
        return img

    @property
    def labels(self):
        """[(crop, health), ...] - 샘플러/가중치 계산에 사용"""
        return [(c, h) for _, c, h in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, crop, health = self.items[idx]
        image = self.images[idx] if self.images is not None else self._open(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, crop, health


class RandomRot90:
    """0/90/180/270도 중 하나로 무작위 회전 (정사각형 이미지라 정보 손실이 없다)."""

    _OPS = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]

    def __call__(self, img):
        op = self._OPS[int(torch.randint(0, 4, (1,)))]
        return img if op is None else img.transpose(op)


def get_transforms(train, aug="basic"):
    """이미지는 이미 128x128 로 맞춰져 있으므로 Resize 는 하지 않는다.

    평가용(train=False) 전처리는 채점 서버와 같아야 하므로 절대 바꾸지 않는다.

    aug (학습용 증강 강도)
      basic  : 좌우/상하 뒤집기 (기존)
      medium : basic + 90도 회전 + 약한 회전/이동/확대 + 약한 색 변화
      strong : medium + RandomErasing (작은 조각 가리기)
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if not train:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    if aug == "basic":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    # 잎 사진은 방향이 제각각이라 회전/뒤집기에 안전하다.
    # 병반(갈색/노란 점)의 색이 건강 상태 단서라서 색 변화(특히 hue)는 아주 약하게만 준다.
    # 회전으로 생기는 빈 모서리는 회색(128)으로 채운다 (정규화하면 0 = 배경 회색과 비슷).
    ops = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRot90(),
        transforms.RandomAffine(degrees=20, translate=(0.06, 0.06), scale=(0.9, 1.1),
                                fill=(128, 128, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        normalize,
    ]
    if aug == "strong":
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0))
    elif aug != "medium":
        raise ValueError(f"unknown aug: {aug}")
    return transforms.Compose(ops)
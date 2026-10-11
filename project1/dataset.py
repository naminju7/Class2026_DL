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


def get_transforms(train, aug="basic", blur_p=0.0):
    """이미지는 이미 128x128 로 맞춰져 있으므로 Resize 는 하지 않는다.

    평가용(train=False) 전처리는 채점 서버와 같아야 하므로 절대 바꾸지 않는다.
    blur_p > 0 이면 (mild/medium/strong 에서) 그 확률로 약한 GaussianBlur 를 적용한다.

    aug (학습용 증강 강도)
      basic  : 좌우/상하 뒤집기 (A 실험)
      mild   : basic + 90도 회전 + (50% 확률로) ±10도/이동 4%/확대 ±5% + 아주 약한 밝기/대비/채도 변화 (hue 없음)
      medium : basic + 90도 회전 + ±20도/이동 6%/확대 ±10% + 약한 색 변화
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

    if aug not in ("mild", "medium", "strong"):
        raise ValueError(f"unknown aug: {aug}")

    # 잎 사진은 방향이 제각각이라 회전/뒤집기에 안전하다.
    # 병반(갈색/노란 점)의 색이 건강 상태 단서라서 색 변화(특히 hue)는 아주 약하게만 준다.
    # 회전으로 생기는 빈 모서리는 회색(128)으로 채운다 (정규화하면 0 = 배경 회색과 비슷).
    bilinear = transforms.InterpolationMode.BILINEAR
    if aug == "mild":
        # 90도 회전/뒤집기는 픽셀을 그대로 옮기므로 무손실. 작은 affine 은 보간으로 살짝 부드러워지기 때문에
        # 절반의 샘플에만 적용해서 원본처럼 선명한 샘플도 계속 보게 한다 (평가 이미지는 보간이 없다).
        affine = transforms.RandomApply([
            transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.95, 1.05),
                                    interpolation=bilinear, fill=(128, 128, 128)),
        ], p=0.5)
        jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0)
    else:
        affine = transforms.RandomAffine(degrees=20, translate=(0.06, 0.06), scale=(0.9, 1.1),
                                         interpolation=bilinear, fill=(128, 128, 128))
        jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)
    ops = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRot90(),
        affine,
        jitter,
    ]
    if blur_p > 0:
        # 초점/해상도 차이에 대한 약한 대비 (기본 꺼짐, 우선순위 낮은 후보)
        ops.append(transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.8))], p=blur_p))
    ops += [transforms.ToTensor(), normalize]
    if aug == "strong":
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0))
    return transforms.Compose(ops)
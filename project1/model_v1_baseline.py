import torch
import torch.nn as nn


class MultiTaskCNN(nn.Module):
    def __init__(self, num_crops=3, num_health=2):
        super().__init__()

        # ----- 특징 추출부 -----
        # baseline보다 채널 수와 층 수를 늘리고 BatchNorm을 추가
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),      # 128 -> 64
            block(32, 64),     # 64 -> 32
            block(64, 128),    # 32 -> 16
            block(128, 256),   # 16 -> 8
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)

        # ----- 두 개의 출력 머리 -----
        self.crop_head = nn.Linear(256, num_crops)
        self.health_head = nn.Linear(256, num_health)

    def forward(self, x):
        # x: (B, 3, 128, 128)
        x = self.features(x)          # (B, 256, 8, 8)
        x = self.pool(x)              # (B, 256, 1, 1)
        feat = torch.flatten(x, 1)    # (B, 256)
        feat = self.dropout(feat)

        crop_out = self.crop_head(feat)      # (B, 3)
        health_out = self.health_head(feat)  # (B, 2)
        return crop_out, health_out
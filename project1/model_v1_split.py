import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """과제별 작은 MLP head. out_features 속성은 train/check 코드 호환용."""

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskCNN(nn.Module):
    def __init__(self, num_crops=3, num_health=2):
        super().__init__()

        # R3/model_v1_baseline과 동일한 backbone
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

        # backbone/GAP은 그대로 두고 head만 과제별 MLP로 분리
        self.crop_head = TaskHead(256, 128, num_crops)
        self.health_head = TaskHead(256, 128, num_health)

    def forward(self, x):
        # x: (B, 3, 128, 128)
        x = self.features(x)          # (B, 256, 8, 8)
        x = self.pool(x)              # (B, 256, 1, 1)
        feat = torch.flatten(x, 1)    # (B, 256)

        crop_out = self.crop_head(feat)      # (B, 3)
        health_out = self.health_head(feat)  # (B, 2)
        return crop_out, health_out

import torch
import torch.nn as nn

# model_v1_baseline.py 와 파라미터/버퍼/state_dict 키가 완전히 같은 모델 (그래서 A.pt 를 그대로 load 할 수 있다).
# 다른 점은 forward 뿐이다: model.eval() 상태에서만, 뒤집은 입력의 logits 를 평균낸다 (학습 중에는 영향 없음).
#
#   tta_mode = "none" : 원본만 (v1 과 똑같이 동작)
#              "h"    : 원본 + 좌우 뒤집기
#              "v"    : 원본 + 상하 뒤집기
#              "hv"   : 원본 + 좌우 + 상하 + 좌우상하 (4뷰)
DEFAULT_TTA = "hv"
TTA_VIEWS = {
    "none": [None],
    "h": [None, (3,)],
    "v": [None, (2,)],
    "hv": [None, (3,), (2,), (2, 3)],
}


class MultiTaskCNN(nn.Module):
    def __init__(self, num_crops=3, num_health=2, tta_mode=DEFAULT_TTA):
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

        # 문자열 속성이라 state_dict 에는 들어가지 않는다
        self.tta_mode = tta_mode

    def _forward(self, x):
        x = self.features(x)          # (B, 256, 8, 8)
        x = self.pool(x)              # (B, 256, 1, 1)
        feat = torch.flatten(x, 1)    # (B, 256)
        feat = self.dropout(feat)
        return self.crop_head(feat), self.health_head(feat)   # (B, 3), (B, 2)

    def forward(self, x):
        # x: (B, 3, 128, 128)
        if self.training or self.tta_mode == "none":
            return self._forward(x)
        views = TTA_VIEWS[self.tta_mode]
        crop_sum, health_sum = 0, 0
        for dims in views:
            c, h = self._forward(x if dims is None else x.flip(dims))
            crop_sum = crop_sum + c
            health_sum = health_sum + h
        return crop_sum / len(views), health_sum / len(views)
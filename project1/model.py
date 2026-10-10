import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    """Squeeze-and-Excitation: 채널별 중요도를 학습해 곱해 주는 가벼운 attention."""

    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(8, ch // reduction)
        self.fc1 = nn.Conv2d(ch, mid, kernel_size=1)
        self.fc2 = nn.Conv2d(mid, ch, kernel_size=1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s


class ResBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN-SE + shortcut(skip connection) -> ReLU.
    stride=2 이면 해상도를 절반으로 줄인다 (MaxPool 대신 stride 사용).
    k2=1 이면 두 번째 conv 를 1x1 로 써서 파라미터를 아낀다."""

    def __init__(self, in_ch, out_ch, stride=1, k2=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k2, stride=1, padding=k2 // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SE(out_ch)
        if stride != 1 or in_ch != out_ch:
            layers = [nn.AvgPool2d(stride)] if stride != 1 else []
            layers += [nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
            self.shortcut = nn.Sequential(*layers)
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.act(out + self.shortcut(x))


class Head(nn.Module):
    """출력 머리: Dropout -> Linear -> ReLU -> Dropout -> Linear.
    공유 특징 위에서 과제별로 한 번 더 변환하므로 두 과제가 서로 덜 간섭한다.
    (out_features 속성은 train.py / 채점 코드 호환용)"""

    def __init__(self, in_features, hidden, out_features, drop):
        super().__init__()
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskCNN(nn.Module):
    def __init__(self, num_crops=3, num_health=2, tta=True):
        super().__init__()
        # tta=True 이면 model.eval() 상태에서만 원본 + 180도 뒤집은(상하좌우) 입력 2장의 logits 평균을 반환한다.
        # (학습 중에는 영향 없음, 파라미터/state_dict 변화 없음. 채점 시간이 걱정되면 tta=False 로)
        self.tta = tta

        # ----- 특징 추출부 (공유) -----
        self.stem = nn.Sequential(                       # 128 -> 128
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResBlock(32, 48, stride=2),                  # 128 -> 64
            ResBlock(48, 48),
            ResBlock(48, 96, stride=2),                  # 64 -> 32
            ResBlock(96, 96),
            ResBlock(96, 192, stride=2),                 # 32 -> 16
            ResBlock(192, 256, stride=2, k2=1),          # 16 -> 8
        )

        # global average pooling 은 병변처럼 "작지만 강한" 신호를 희석시키므로 max pooling 도 같이 쓴다
        feat_dim = 256 * 2

        # ----- 두 개의 출력 머리 (과제별로 분리) -----
        self.crop_head = Head(feat_dim, 128, num_crops, drop=0.3)
        self.health_head = Head(feat_dim, 128, num_health, drop=0.3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 각 residual block 의 마지막 BN 을 0 으로 시작 -> 처음에는 항등 함수에 가까워 학습이 안정적
        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.bn2.weight)

    def _forward(self, x):
        x = self.features(self.stem(x))                   # (B, 256, 8, 8)
        feat = torch.cat([F.adaptive_avg_pool2d(x, 1),
                          F.adaptive_max_pool2d(x, 1)], dim=1).flatten(1)  # (B, 512)
        return self.crop_head(feat), self.health_head(feat)

    def forward(self, x):
        # x: (B, 3, 128, 128) -> crop_logits (B, 3), health_logits (B, 2)
        if self.tta and not self.training:
            # 4장([None, (3,), (2,), (2, 3)])으로 늘리면 조금 더 안정적이지만 채점 시간이 2배가 된다
            flips = [None, (2, 3)]
            crop_sum, health_sum = 0, 0
            for dims in flips:
                c, h = self._forward(x if dims is None else x.flip(dims))
                crop_sum = crop_sum + c
                health_sum = health_sum + h
            return crop_sum / len(flips), health_sum / len(flips)
        return self._forward(x)
import torch.nn as nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class WaferCNN(nn.Module):
    """64x64 입력 기준 작은 CNN.

    ResNet18 전이학습도 해봤는데 웨이퍼맵이 자연 이미지랑 너무 달라서
    (3값짜리 저해상도 맵) 큰 이득이 없었고, 파라미터만 20배 컸음.
    그래서 그냥 처음부터 학습하는 작은 모델로 확정.
    """

    def __init__(self, num_classes=9, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_ch, 32),   # 64 -> 32
            conv_block(32, 64),      # 32 -> 16
            conv_block(64, 128),     # 16 -> 8
            conv_block(128, 128),    # 8 -> 4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

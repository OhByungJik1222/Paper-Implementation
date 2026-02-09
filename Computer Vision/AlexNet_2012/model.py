import torch
import torch.nn as nn

# ReLU 활성화 함수 (Section 3.1) - 학습 시간 단축
# LRN (Section 3.3) - 과적합 방지
# Overlapping Pooling (Section 3.4) - 과적합 방지
# Dropout (Section 4.2) - 과적합 방지
# Bias Initialization (Section 5) - ReLU 함수에 양의 입력 -> 초기 단계 학습 가속
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000): # Section 3.5
        # 입력 크기: (b x 3 x 227 x 227)
        # 논문 상에선 입력 크기가 224라고 작성되어 있지만 실제론 227
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), # (b x 96 x 55 x 55)
            nn.ReLU(), # Section 3.1
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), # Section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2), # Section 3.4 (b x 96 x 27 x 27)
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2), # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), # (b x 256 x 13 x 13)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1), # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1), # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1), # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2) # (b x 256 x 6 x 6)
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5), # Section 4.2
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        self.init_wb()

    def init_wb(self): # Section 5
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.conv[4].bias, 1)
        nn.init.constant_(self.conv[10].bias, 1)
        nn.init.constant_(self.conv[12].bias, 1)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
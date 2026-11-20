"""SqueezeNet architecture adapted for 32x32 CIFAR inputs.

Built from the Fire-module design (Iandola et al.) with a 3x3 stem and no aggressive
initial downsampling, so useful spatial resolution survives to the classifier.
"""
import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1, expand3x3):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, 1, bias=False),
            nn.BatchNorm2d(squeeze_channels), nn.ReLU(inplace=True))
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand1x1, 1, bias=False),
            nn.BatchNorm2d(expand1x1), nn.ReLU(inplace=True))
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand3x3, 3, padding=1, bias=False),
            nn.BatchNorm2d(expand3x3), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], dim=1)


class SqueezeNetCIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 16x16
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(2),                                   # 8x8
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(2),                                   # 4x4
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.features.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.classifier.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.features(x)).flatten(1)

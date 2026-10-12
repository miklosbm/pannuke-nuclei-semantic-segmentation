import torch
import torch.nn as nn


# SimpleUNet + Dropout2d in each decoder block, to reduce overfitting
class SimpleUNet2(nn.Module):
    def __init__(self, num_classes=6, dropout=0.2):
        super().__init__()

        # encoder stage 1: 3 -> 32 channels
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        # encoder stage 2: 32 -> 64 channels
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        # encoder stage 3: 64 -> 128 channels
        self.enc3_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        # halves spatial resolution between encoder stages
        self.pool = nn.MaxPool2d(2, 2)

        # decoder stage 1: upsample + fuse with encoder stage 3, then dropout
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        # decoder stage 2: fuse with encoder stage 2, then dropout
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        # decoder stage 3: fuse with encoder stage 1, then dropout
        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        # 1x1 conv maps 32 channels -> per-pixel class scores
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # same encoder/decoder/skip-connection flow as SimpleUNet
        e1 = self.enc1_conv(x)
        x = self.pool(e1)

        e2 = self.enc2_conv(x)
        x = self.pool(e2)

        e3 = self.enc3_conv(x)
        x = self.pool(e3)

        x = self.up1(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec1_conv(x)

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2_conv(x)

        x = self.up3(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec3_conv(x)

        return self.final(x)

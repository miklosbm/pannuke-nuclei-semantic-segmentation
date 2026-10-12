import torch
import torch.nn as nn


# baseline U-Net: 3 encoder stages + 3 decoder stages with skip connections
class SimpleUNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # encoder stage 1: 3 -> 32 channels, full resolution
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

        # decoder stage 1: upsample the bottleneck, then fuse with encoder stage 3
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )

        # decoder stage 2: fuse with encoder stage 2
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
        )

        # decoder stage 3: fuse with encoder stage 1
        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
        )

        # 1x1 conv maps 32 channels -> per-pixel class scores
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # encoder path, keeping each stage's output for its skip connection
        e1 = self.enc1_conv(x)      # teljes felbontas, 32ch  <- elmentve
        x = self.pool(e1)

        e2 = self.enc2_conv(x)      # 1/2 felbontas, 64ch     <- elmentve
        x = self.pool(e2)

        e3 = self.enc3_conv(x)      # 1/4 felbontas, 128ch    <- elmentve
        x = self.pool(e3)           # 1/8 felbontas (bottleneck)

        # decoder path: upsample, then concatenate with the matching encoder skip
        x = self.up1(x)
        x = torch.cat([x, e3], dim=1)   # skip connection
        x = self.dec1_conv(x)

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)   # skip connection
        x = self.dec2_conv(x)

        x = self.up3(x)
        x = torch.cat([x, e1], dim=1)   # skip connection
        x = self.dec3_conv(x)

        return self.final(x)

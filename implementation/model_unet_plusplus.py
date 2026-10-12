import torch
import torch.nn as nn


class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=6, dropout=0.2):
        super().__init__()

        # A basic building block: Conv → BatchNorm → ReLU → (Dropout)
        # in_ch  = how many feature maps come in
        # out_ch = how many feature maps come out
        def conv_block(in_ch, out_ch, use_dropout=False):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),  # extract features
                nn.BatchNorm2d(out_ch),                               # stabilize training
                nn.ReLU(inplace=True),                                # add non-linearity
            ]
            if use_dropout:
                layers.append(nn.Dropout2d(dropout))                  # prevent overfitting
            return nn.Sequential(*layers)

        # ENCODER
        # Each block extracts features, then pool() halves the resolution
        # 256x256 → 128x128 → 64x64 → 32x32
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = conv_block(3,    32)    # sees fine details (edges)
        self.enc2 = conv_block(32,   64)    # sees local patterns (cell borders)
        self.enc3 = conv_block(64,  128)    # sees larger structures
        self.enc4 = conv_block(128, 256)    # sees global context

        # UPSAMPLERS
        # ConvTranspose2d doubles the resolution (opposite of pool)
        # We need one for every upward arrow in the grid below
        #
        #  enc1 → node1_1 → node1_2 → node1_3 → output
        #    ↑        ↑          ↑          ↑
        #  enc2 → node2_1 → node2_2
        #    ↑        ↑          ↑
        #  enc3 → node3_1
        #    ↑        ↑
        #  enc4
        #
        self.up_2_to_1   = nn.ConvTranspose2d(64,  64,  kernel_size=2, stride=2)  # 128→256
        self.up_3_to_2   = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  #  64→128
        self.up_4_to_3   = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  #  32→ 64
        self.up_21_to_11 = nn.ConvTranspose2d(64,  64,  kernel_size=2, stride=2)  # 128→256
        self.up_31_to_21 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  #  64→128
        self.up_22_to_12 = nn.ConvTranspose2d(64,  64,  kernel_size=2, stride=2)  # 128→256

        # NODES
        # Each node receives multiple inputs via torch.cat()
        # in_ch = sum of all input channels
        #
        # Column 1: each node gets enc + upsampled deeper level
        self.node1_1 = conv_block(32  + 64,        32, use_dropout=True)   # enc1(32)  + up(enc2→64)        = 96
        self.node2_1 = conv_block(64  + 128,       64, use_dropout=True)   # enc2(64)  + up(enc3→128)       = 192
        self.node3_1 = conv_block(128 + 256,       128, use_dropout=True)  # enc3(128) + up(enc4→256)       = 384
        #
        # Column 2: each node gets enc + previous node + upsampled deeper node
        self.node1_2 = conv_block(32  + 32  + 64,  32, use_dropout=True)   # enc1(32)  + node1_1(32) + up(node2_1→64)  = 128
        self.node2_2 = conv_block(64  + 64  + 128, 64, use_dropout=True)   # enc2(64)  + node2_1(64) + up(node3_1→128) = 256
        #
        # Column 3: node gets enc + all previous nodes + upsampled deeper node
        self.node1_3 = conv_block(32 + 32 + 32 + 64, 32, use_dropout=True)  # enc1(32) + node1_1(32) + node1_2(32) + up(node2_2→64) = 160

        # OUTPUT
        # 1x1 conv maps 32 feature channels → num_classes scores per pixel
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):

        # ENCODER — extract features at 4 scales
        x1_0 = self.enc1(x)                # 32ch,  256x256
        x2_0 = self.enc2(self.pool(x1_0))  # 64ch,  128x128
        x3_0 = self.enc3(self.pool(x2_0))  # 128ch,  64x64
        x4_0 = self.enc4(self.pool(x3_0))  # 256ch,  32x32

        # COLUMN 1 — first refinement
        # merge same-level encoder + upsampled deeper encoder
        x1_1 = self.node1_1(torch.cat([x1_0,             self.up_2_to_1(x2_0)],   dim=1))
        x2_1 = self.node2_1(torch.cat([x2_0,             self.up_3_to_2(x3_0)],   dim=1))
        x3_1 = self.node3_1(torch.cat([x3_0,             self.up_4_to_3(x4_0)],   dim=1))

        # COLUMN 2 — second refinement
        # merge same-level encoder + column1 node + upsampled deeper column1 node
        x1_2 = self.node1_2(torch.cat([x1_0, x1_1,       self.up_21_to_11(x2_1)], dim=1))
        x2_2 = self.node2_2(torch.cat([x2_0, x2_1,       self.up_31_to_21(x3_1)], dim=1))

        # COLUMN 3 — third refinement
        # merge same-level encoder + all column1,2 nodes + upsampled deeper column2 node
        x1_3 = self.node1_3(torch.cat([x1_0, x1_1, x1_2, self.up_22_to_12(x2_2)], dim=1))

        # OUTPUT — predict class for every pixel
        return self.final(x1_3)

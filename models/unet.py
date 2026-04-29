import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.d1 = DoubleConv(in_ch, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.u3 = DoubleConv(512+256, 256)
        self.u2 = DoubleConv(256+128, 128)
        self.u1 = DoubleConv(128+64, 64)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))

        x = self.up(c4)
        x = self.u3(torch.cat([x, c3], 1))
        x = self.up(x)
        x = self.u2(torch.cat([x, c2], 1))
        x = self.up(x)
        x = self.u1(torch.cat([x, c1], 1))

        return torch.sigmoid(self.out(x))

# models/unet.py
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f
        self.bottleneck = DoubleConv(in_ch, in_ch*2)
        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(in_ch*2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(in_channels=f*2, out_channels=f))
            in_ch = f
        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)
    def forward(self, x):
        skip_connections = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for upconv, up in zip(self.upconvs, self.ups):
            x = upconv(x)
            skip = skip_connections.pop(0)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = up(x)
        return torch.sigmoid(self.final(x))

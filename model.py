import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: Number of channels in the gating signal (decoder features)
        F_l: Number of channels in the encoder features
        F_int: Intermediate channels for attention computation
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class SimpleUNetWithAttention(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(SimpleUNetWithAttention, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Attention Gates
        self.ag1 = AttentionGate(F_g=256, F_l=128, F_int=64)  # For e2
        self.ag2 = AttentionGate(F_g=128, F_l=64, F_int=32)  # For e1

        # Decoder
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)  # [B, 64, H, W]
        e2 = self.enc2(e1)  # [B, 128, H/2, W/2]
        e3 = self.enc3(e2)  # [B, 256, H/4, W/4]

        # Decoder path with attention
        d1 = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 256, H/2, W/2]
        attended_e2 = self.ag1(g=d1, x=e2)  # [B, 128, H/2, W/2]
        d1 = torch.cat((attended_e2, d1), dim=1)  # [B, 384, H/2, W/2]
        d1 = self.dec_conv1(d1)  # [B, 128, H/2, W/2]

        d2 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 128, H, W]
        attended_e1 = self.ag2(g=d2, x=e1)  # [B, 64, H, W]
        d2 = torch.cat((attended_e1, d2), dim=1)  # [B, 192, H, W]
        d2 = self.dec_conv2(d2)  # [B, 64, H, W]

        out = self.final_conv(d2)  # [B, 1, H, W]

        out = F.interpolate(out, size=(256, 512), mode='bilinear', align_corners=True)  # [B, 1, 256, 512]
        return out
from __future__ import annotations

import torch
from torch import nn


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.0,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout and dropout > 0 else nn.Identity()
        if use_residual and in_ch != out_ch:
            self.residual_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        if self.use_residual:
            out = out + self.residual_conv(x)
        return self.relu(out)


class AttentionGate3D(nn.Module):
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv3d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_ch),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv3d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.gate_conv(gate)
        x = self.skip_conv(skip)
        attn = self.relu(g + x)
        alpha = self.psi(attn)
        return skip * alpha


class AttentionUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 16,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock3D(
            in_channels,
            base_channels,
            dropout=dropout,
            use_residual=use_residual,
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock3D(
            base_channels,
            base_channels * 2,
            dropout=dropout,
            use_residual=use_residual,
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock3D(
            base_channels * 2,
            base_channels * 4,
            dropout=dropout,
            use_residual=use_residual,
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock3D(
            base_channels * 4,
            base_channels * 8,
            dropout=dropout,
            use_residual=use_residual,
        )

        # Decoder
        self.up3 = nn.ConvTranspose3d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.att3 = AttentionGate3D(
            gate_ch=base_channels * 4,
            skip_ch=base_channels * 4,
            inter_ch=base_channels * 2,
        )
        self.dec3 = ConvBlock3D(
            base_channels * 8,
            base_channels * 4,
            dropout=dropout,
            use_residual=use_residual,
        )

        self.up2 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.att2 = AttentionGate3D(
            gate_ch=base_channels * 2,
            skip_ch=base_channels * 2,
            inter_ch=base_channels,
        )
        self.dec2 = ConvBlock3D(
            base_channels * 4,
            base_channels * 2,
            dropout=dropout,
            use_residual=use_residual,
        )

        self.up1 = nn.ConvTranspose3d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.att1 = AttentionGate3D(
            gate_ch=base_channels,
            skip_ch=base_channels,
            inter_ch=base_channels // 2,
        )
        self.dec1 = ConvBlock3D(
            base_channels * 2,
            base_channels,
            dropout=dropout,
            use_residual=use_residual,
        )

        self.out_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Decoder with attention gates on skip connections
        d3 = self.up3(e4)
        s3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up2(d3)
        s2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)
        s1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.out_conv(d1)

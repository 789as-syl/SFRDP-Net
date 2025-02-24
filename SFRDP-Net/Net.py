import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class FourierUnit(nn.Module):
    def __init__(self, dim):
        super(FourierUnit, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1), nn.ReLU(True),
                                  nn.Conv2d(dim * 2, dim * 2, 1))

    def forward(self, x):
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim)
        ffted = torch.cat((ffted.real, ffted.imag), 1)
        ffted = self.conv(ffted)
        real, imag = torch.chunk(ffted, 2, dim=1)
        ffted = torch.complex(real, imag)
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim)
        return output

class Att(nn.Module):
    def __init__(self, dim):
        super(Att, self).__init__()
        self.four = FourierUnit(dim)
        self.conv1 = nn.Conv2d(dim * 2, dim, 1)
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.gate = Gate(dim)
        self.softmax = nn.Softmax(dim=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, bias=True)
        )
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect', groups=dim, bias=True),
            nn.Conv2d(dim, dim // 8, 1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(dim // 8, dim, 1, padding=0),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect', groups=dim, bias=True),
        )

    def forward(self, x, y):
        res = self.conv1(torch.cat([x, y], 1))
        att1 = self.mlp(self.avg_pool(res)).expand_as(res) + self.mlp(self.max_pool(res)).expand_as(res) + self.pa(res)
        att2 = self.four(res)
        att = self.gate(att2, att1)
        att = self.conv2(att)
        out = att * res
        return out


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=True, padding_mode='reflect'), nn.ReLU(True),
                                  nn.Conv2d(dim, dim, 3, 1, 1, bias=True, padding_mode='reflect'))

    def forward(self, x):
        return self.conv(x) + x


class Gate(nn.Module):
    def __init__(self, dim):
        super(Gate, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def forward(self, x, y):
        res = self.conv(x - y) * x + y
        return res


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class FFC_ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(FFC_ResnetBlock, self).__init__()
        self.ffc1 = FourierUnit(dim)
        self.ffc2 = FourierUnit(dim)
        self.res1 = ResBlock(dim)
        self.res2 = ResBlock(dim)
        self.att = Att(dim)
        self.cat = nn.Conv2d(dim * 2, dim, 1)
        self.g1 = Gate(dim)
        self.g2 = Gate(dim)

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.ffc1(x)
        res1 = self.g1(x2, x1)
        res2 = self.g2(x1, x2)
        res1 = self.res2(res1)
        res2 = self.ffc2(res2)
        out = self.att(res1, res2) + x
        return out


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class Net(nn.Module):
    def __init__(self, dim=24):
        super(Net, self).__init__()
        self.in_conv = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(3, dim, kernel_size=7, padding=0))
        self.down1 = nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, padding_mode='reflect'))

        self.down2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1, padding_mode='reflect'))

        self.up2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2 * 4, kernel_size=1,
                      padding=0, padding_mode='reflect'),
            nn.PixelShuffle(2)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=1,
                      padding=0, padding_mode='reflect'),
            nn.PixelShuffle(2)
        )

        self.conv = nn.Sequential(nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))

        blocks1 = [FFC_ResnetBlock(dim) for _ in range(2)]
        self.g1 = nn.Sequential(*blocks1)

        blocks2 = [FFC_ResnetBlock(dim * 2) for _ in range(2)]
        self.g2 = nn.Sequential(*blocks2)

        blocks3 = [FFC_ResnetBlock(dim * 4) for _ in range(4)]
        self.g3 = nn.Sequential(*blocks3)

        blocks4 = [FFC_ResnetBlock(dim * 2) for _ in range(2)]
        self.g4 = nn.Sequential(*blocks4)

        blocks5 = [FFC_ResnetBlock(dim) for _ in range(2)]
        self.g5 = nn.Sequential(*blocks5)

        self.mix1 = SKFusion(dim * 2)
        self.mix2 = SKFusion(dim)

    def forward(self, x):
        res1 = self.g1(self.in_conv(x))
        res2 = self.g2(self.down1(res1))
        res3 = self.g3(self.down2(res2))
        res4 = self.g4(self.mix1([self.up2(res3), res2]))
        res5 = self.g5(self.mix2([self.up3(res4), res1]))
        out = self.conv(res5)
        return out + x



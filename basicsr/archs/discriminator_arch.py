from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torch
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.continuous_arch import LayerNorm2d


@ARCH_REGISTRY.register()
class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch, num_feat, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        assert self.input_size == 128 or self.input_size == 256, (
            f'input size must be 128 or 256, but received {input_size}')

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

        if self.input_size == 256:
            feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
            feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


@ARCH_REGISTRY.register(suffix='basicsr')
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

class StereoInfoMerge(nn.Module):
    def __init__(self,num_feat):
        super(StereoInfoMerge, self).__init__()
        self.norm_l = LayerNorm2d(num_feat)
        self.norm_r = LayerNorm2d(num_feat)
        self.l_proj1 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0)

        self.l_proj2 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0)

        self.merge_conv = nn.Conv2d(num_feat, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T)
        F_r2l = torch.matmul(torch.softmax((attention), dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax((attention).permute(0, 1, 3, 2), dim=-1),
                             V_l)  # B, H, W, c
        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2)
        F_l2r = F_l2r.permute(0, 3, 1, 2)
        return self.merge_conv(F_r2l * F_l2r)


@ARCH_REGISTRY.register(suffix='basicsr')
class StereoDiscriminatorSN(nn.Module):
    """Defines a Stereo U-Net discriminator with spectral normalization (SN)


    Arg:
        num_in_ch (int): Channel number of inputs. Default: 6.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch=3, num_feat=32, skip_connection=True):
        super(StereoDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))

        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))

        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = nn.Conv2d(num_feat * 2, 6, 3, 1, 1)
        self.conv9 = nn.Conv2d(9, 1, 3, 1, 1)
        self.stereo_info1 = StereoInfoMerge(num_feat)
        self.stereo_info2 = StereoInfoMerge(num_feat * 2)
        self.stereo_info3 = StereoInfoMerge(num_feat * 4)
    def forward(self, x):
        # downsample
        x00, x01 = F.leaky_relu(self.conv0(x[:,:3,:,:]), negative_slope=0.2, inplace=True),F.leaky_relu(self.conv0(x[:,3:,:,:]), negative_slope=0.2, inplace=True)
        x10, x11 = F.leaky_relu(self.conv1(x00), negative_slope=0.2, inplace=True),F.leaky_relu(self.conv1(x01), negative_slope=0.2, inplace=True)
        x20, x21 = F.leaky_relu(self.conv2(x10), negative_slope=0.2, inplace=True), F.leaky_relu(self.conv2(x11), negative_slope=0.2, inplace=True)
        x30, x31 = F.leaky_relu(self.conv3(x20), negative_slope=0.2, inplace=True), F.leaky_relu(self.conv3(x21), negative_slope=0.2, inplace=True)

        # upsample
        x30, x31 = F.interpolate(x30, scale_factor=2, mode='bilinear', align_corners=False),F.interpolate(x31, scale_factor=2, mode='bilinear', align_corners=False)
        x40, x41 = F.leaky_relu(self.conv4(x30), negative_slope=0.2, inplace=True),F.leaky_relu(self.conv4(x31), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x40 = x40 + x20
            x41 = x41 + x21
        mf3 = self.stereo_info3(x40, x41)
        x40, x41 = F.interpolate(x40, scale_factor=2, mode='bilinear', align_corners=False),F.interpolate(x41, scale_factor=2, mode='bilinear', align_corners=False)
        x50, x51 = F.leaky_relu(self.conv5(x40), negative_slope=0.2, inplace=True),F.leaky_relu(self.conv5(x41), negative_slope=0.2, inplace=True)
        mf2 = self.stereo_info2(x50, x51)
        if self.skip_connection:
            x50 = x50 + x10
            x51 = x51 + x11
        x50, x51 = F.interpolate(x50, scale_factor=2, mode='bilinear', align_corners=False), F.interpolate(x51, scale_factor=2, mode='bilinear', align_corners=False)
        x60, x61 = F.leaky_relu(self.conv6(x50), negative_slope=0.2, inplace=True), F.leaky_relu(self.conv6(x51), negative_slope=0.2, inplace=True)
        mf1 = self.stereo_info1(x60, x61)
        if self.skip_connection:
            x60 = x60 + x00
            x61 = x61 + x01
        # extra convolutions
        x70, x71 = F.leaky_relu(self.conv7(x60), negative_slope=0.2, inplace=True), F.leaky_relu(self.conv7(x61), negative_slope=0.2, inplace=True)
        mf0 = self.conv8(torch.cat([x70, x71],dim=1))
        out = self.conv9(torch.cat([mf0, mf1, F.interpolate(mf2,scale_factor=2,mode='bilinear', align_corners=False), F.interpolate(mf3,scale_factor=4,mode='bilinear', align_corners=False)],dim=1))

        return out
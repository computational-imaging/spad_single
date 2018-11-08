import torch
import torch.nn as nn
import numpy as np
import skimage.transform
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor


class _DS_Block(nn.Module):
    def __init__(self):
        super(_DS_Block, self).__init__()

        self.ds_block = nn.Sequential(
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        output = self.ds_block(x)
        return output


class FusionDenoiseModel(nn.Module):
    def __init__(self):
        super(FusionDenoiseModel, self).__init__()
        self.ds1 = nn.Sequential(
            nn.Conv3d(1, 1, 7, stride=2, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds2 = nn.Sequential(
            nn.Conv3d(1, 1, 5, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds3 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(36, 36, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(36),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(28, 28, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(28),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(41, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.regress = nn.Sequential(
            nn.Conv3d(16, 1, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
        )

        self.conv0 = nn.Sequential(
            nn.Conv3d(1, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.intensity_ds = nn.Sequential(
            torch.nn.Conv2d(1, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        self.refine_depth1 = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.refine_depth2 = nn.Sequential(
            torch.nn.Conv2d(33, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids_in = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids1 = _DS_Block()
        self.ids2 = _DS_Block()

        self.iskip = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, (6, 6), (2, 2), 2)
        )

        self.iup1 = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup1_refine = nn.Sequential(
            torch.nn.Conv2d(64, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup2 = nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup2_refine = nn.Sequential(
            torch.nn.Conv2d(48, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup3 = nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
        )
        self.iup3_refine = nn.Sequential(
            torch.nn.Conv2d(40, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        # https://github.com/twtygqyy/pytorch-LapSRN/blob/master/lapsrn.py#L6
        def get_upsample_filter(size):
            """Make a 2D bilinear kernel suitable for upsampling"""
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1.
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size, :size]
            filter = (1 - abs(og[0] - center) / factor) * \
                     (1 - abs(og[1] - center) / factor) * \
                     (1 - abs(og[2] - center) / factor)
            return torch.from_numpy(filter).float()

        for n in [self.up1, self.up2, self.up3, self.ds1, self.ds2, self.ds3]:
            for m in n:
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                    c1, c2, d, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, d, h, w).repeat(c1, c2, 1, 1, 1)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, spad, intensity):

        # downsample intensity image
        intensity_ds_out = intensity
        tiled_intensity_ds_out = intensity_ds_out.repeat(1, spad.size()[2], 1, 1).unsqueeze(1)

        # pass spad through autoencoder
        smax = torch.nn.Softmax2d()

        ds1_out = self.ds1(spad)
        ds2_out = self.ds2(ds1_out)
        ds3_out = self.ds3(ds2_out)

        conv0_out = self.conv0(spad)
        conv1_out = self.conv1(ds1_out)
        conv2_out = self.conv2(ds2_out)
        conv3_out = self.conv3(ds3_out)

        up3_out = self.up3(conv3_out)
        up2_out = self.up2(torch.cat((conv2_out, up3_out), 1))
        up1_out = self.up1(torch.cat((conv1_out, up2_out), 1))
        up0_out = torch.cat((conv0_out, up1_out), 1)

        refine_out = self.refine(torch.cat((tiled_intensity_ds_out, up0_out), 1))
        regress_out = self.regress(refine_out)

        # squeeze and softmax for pixelwise classification loss
        denoise_out = torch.squeeze(regress_out, 1)

        smax_denoise_out = smax(denoise_out)

        # soft argmax
        weights = Variable(torch.linspace(0, 1, steps=spad.size()[2]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax_denoise_out
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        return denoise_out, soft_argmax


class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()

        self.ds1 = nn.Sequential(
            nn.Conv3d(1, 1, 7, stride=2, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds2 = nn.Sequential(
            nn.Conv3d(1, 1, 5, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds3 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(36, 36, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(36),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(28, 28, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(28),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(40, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.regress = nn.Sequential(
            nn.Conv3d(16, 1, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
        )

        self.conv0 = nn.Sequential(
            nn.Conv3d(1, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.intensity_ds = nn.Sequential(
            torch.nn.Conv2d(1, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        self.refine_depth1 = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.refine_depth2 = nn.Sequential(
            torch.nn.Conv2d(33, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids_in = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids1 = _DS_Block()
        self.ids2 = _DS_Block()

        self.iskip = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, (6, 6), (2, 2), 2)
        )

        self.iup1 = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup1_refine = nn.Sequential(
            torch.nn.Conv2d(64, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup2 = nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup2_refine = nn.Sequential(
            torch.nn.Conv2d(48, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup3 = nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
        )
        self.iup3_refine = nn.Sequential(
            torch.nn.Conv2d(40, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        # https://github.com/twtygqyy/pytorch-LapSRN/blob/master/lapsrn.py#L6
        def get_upsample_filter(size):
            """Make a 2D bilinear kernel suitable for upsampling"""
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1.
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size, :size]
            filter = (1 - abs(og[0] - center) / factor) * \
                     (1 - abs(og[1] - center) / factor) * \
                     (1 - abs(og[2] - center) / factor)
            return torch.from_numpy(filter).float()

        for n in [self.up1, self.up2, self.up3, self.ds1, self.ds2, self.ds3]:
            for m in n:
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                    c1, c2, d, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, d, h, w).repeat(c1, c2, 1, 1, 1)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, spad):

        # pass spad through autoencoder
        smax = torch.nn.Softmax2d()

        ds1_out = self.ds1(spad)
        ds2_out = self.ds2(ds1_out)
        ds3_out = self.ds3(ds2_out)

        conv0_out = self.conv0(spad)
        conv1_out = self.conv1(ds1_out)
        conv2_out = self.conv2(ds2_out)
        conv3_out = self.conv3(ds3_out)

        up3_out = self.up3(conv3_out)
        up2_out = self.up2(torch.cat((conv2_out, up3_out), 1))
        up1_out = self.up1(torch.cat((conv1_out, up2_out), 1))
        up0_out = torch.cat((conv0_out, up1_out), 1)

        refine_out = self.refine(up0_out)
        regress_out = self.regress(refine_out)

        # squeeze and softmax for pixelwise classification loss
        denoise_out = torch.squeeze(regress_out, 1)
        smax_denoise_out = smax(denoise_out)

        # soft argmax
        weights = Variable(torch.linspace(0, 1, steps=spad.size()[2]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax_denoise_out
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        return denoise_out, soft_argmax


class Upsampler8x(nn.Module):
    def __init__(self):
        super(Upsampler8x, self).__init__()

        # intensity
        self.conv1_Y = nn.Sequential(
            nn.Conv2d(1, 49, 7, 1, 3, bias=True)
        )
        self.prelu1_Y = nn.Sequential(
            nn.PReLU(49)
        )
        self.conv2_Y = nn.Sequential(
            nn.Conv2d(49, 32, 5, 1, 2, bias=True)
        )
        self.prelu2_Y = nn.Sequential(
            nn.PReLU(32)
        )
        self.conv3_Y = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2, bias=True)
        )
        self.prelu3_Y = nn.Sequential(
            nn.PReLU(32)
        )
        self.pool3_Y = nn.Sequential(
            nn.MaxPool2d(3, stride=2)
        )
        self.conv4_Y = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2, bias=True)
        )
        self.prelu4_Y = nn.Sequential(
            nn.PReLU(32)
        )
        self.pool4_Y = nn.Sequential(
            nn.MaxPool2d(3, stride=2)
        )

        # depth
        self.conv1_D = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 2, bias=True)
        )
        self.prelu1_D = nn.Sequential(
            nn.PReLU(64)
        )
        self.deconv2_D = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, 2, 2, bias=True)
        )
        self.prelu2_D = nn.Sequential(
            nn.PReLU(32)
        )

        # <--- concatenation pool4_Y and deconv2_D --->

        self.conv2a_x4 = nn.Sequential(
            nn.Conv2d(64, 32, 5, 1, 2, bias=True)
        )
        self.prelu2a_x4 = nn.Sequential(
            nn.PReLU(32)
        )
        self.conv2b_x4 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2, bias=True)
        )
        self.prelu2b_x4 = nn.Sequential(
            nn.PReLU(32)
        )
        self.deconv3_x4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 5, 2, 1, bias=True)
        )
        self.prelu3_x4 = nn.Sequential(
            nn.PReLU(32)
        )

        # <--- concatenation pool3_Y and deconv3_x4 --->

        self.conv3a_x8 = nn.Sequential(
            nn.Conv2d(64, 32, 5, 1, 2, bias=True)
        )
        self.prelu3a_x8 = nn.Sequential(
            nn.PReLU(32)
        )
        self.conv3b_x8 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2, bias=True)
        )
        self.prelu3b_x8 = nn.Sequential(
            nn.PReLU(32)
        )
        self.deconv4_x8 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 5, 2, 1, bias=True)
        )
        self.prelu4_x8 = nn.Sequential(
            nn.PReLU(32)
        )

        # <--- concatenation pool2_Y and deconv4_x8 --->

        self.conv4a = nn.Sequential(
            nn.Conv2d(64, 32, 5, 1, 2, bias=True)
        )
        self.prelu4a = nn.Sequential(
            nn.PReLU(32)
        )
        self.conv4b = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2, bias=True)
        )
        self.prelu4b = nn.Sequential(
            nn.PReLU(32)
        )
        self.conv4c = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2, bias=True)
        )
        self.prelu4c = nn.Sequential(
            nn.PReLU(32)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 1, 5, 1, 2, bias=True)
        )
        self.lp_filter = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        )

        # initialize low pass filter
        for n in [self.lp_filter]:
            for m in n:
                if isinstance(m, nn.Conv2d):
                    c1, f, h, w = m.weight.data.size()
                    weight = torch.ones(3, 3) / 9
                    m.weight.data = weight.view(1, 1, h, w).repeat(c1, f, 1, 1)
                    if m.bias is not None:
                        m.bias.data.zero_()

        # set requires gradient to false
        for f in self.lp_filter.parameters():
            f.requires_grad = False

    def forward(self, depth, intensity):

        intensity = intensity[:, :, 0:-1, 0:-1]

        # prep intensity image
        lp_intensity = self.lp_filter(intensity)
        hp_intensity = intensity - lp_intensity
        for i in range(hp_intensity.size()[0]):
            hp_intensity[i, :, :, :] = (hp_intensity[i, :, :, :] - torch.min(hp_intensity[i, :, :, :])) / \
                (torch.max(hp_intensity[i, :, :, :]) - torch.min(hp_intensity[i, :, :, :]))

        # prep depth image
        min_depth = Variable(torch.zeros(depth.size()[0]).type(dtype))
        max_depth = Variable(torch.zeros(depth.size()[0]).type(dtype))
        depth_normalized = Variable(torch.zeros(depth.size()).type(dtype))
        for i in range(depth.size()[0]):
            min_depth[i] = torch.min(depth[i, :, :, :])
            max_depth[i] = torch.max(depth[i, :, :, :])
            depth_normalized[i, :, :, :] = (depth[i, :, :, :] - min_depth[i]) / (max_depth[i] - min_depth[i])

        lp_depth = self.lp_filter(depth_normalized)
        hp_depth = depth_normalized - lp_depth

        # get high-resolution lp_depth
        lp_depth_np = lp_depth.data.cpu().numpy()
        ups_lp_depth_np = np.zeros((lp_depth_np.shape[0], lp_depth_np.shape[2]*8, lp_depth_np.shape[3]*8))
        for i in range(lp_depth_np.shape[0]):
            ups_lp_depth_np[i, :, :] = skimage.transform.rescale(np.squeeze(lp_depth_np[i, :, :, :]), 8, order=3, mode='symmetric', clip=False)

        ups_lp_depth = Variable(torch.from_numpy(ups_lp_depth_np).unsqueeze(1).type(dtype))
        ups_lp_depth.requires_grad = False

        # pass through intensity network
        conv1_Y_out = self.conv1_Y(hp_intensity)
        prelu1_Y_out = self.prelu1_Y(conv1_Y_out)
        conv2_Y_out = self.conv2_Y(prelu1_Y_out)
        prelu2_Y_out = self.prelu2_Y(conv2_Y_out)
        conv3_Y_out = self.conv3_Y(prelu2_Y_out)
        prelu3_Y_out = self.prelu3_Y(conv3_Y_out)
        pool3_Y_out = self.pool3_Y(prelu3_Y_out)
        conv4_Y_out = self.conv4_Y(pool3_Y_out)
        prelu4_Y_out = self.prelu4_Y(conv4_Y_out)
        pool4_Y_out = self.pool4_Y(prelu4_Y_out)

        # depth network
        conv1_D_out = self.conv1_D(hp_depth)
        prelu1_D_out = self.prelu1_D(conv1_D_out)
        deconv2_D_out = self.deconv2_D(prelu1_D_out)
        prelu2_D_out = self.prelu2_D(deconv2_D_out)

        conv2a_x4_out = self.conv2a_x4(torch.cat((pool4_Y_out, prelu2_D_out), 1))
        prelu2a_x4_out = self.prelu2a_x4(conv2a_x4_out)
        conv2b_x4_out = self.conv2b_x4(prelu2a_x4_out)
        prelu2b_x4_out = self.prelu2b_x4(conv2b_x4_out)
        deconv3_x4_out = self.deconv3_x4(prelu2b_x4_out)
        prelu3_x4_out = self.prelu3_x4(deconv3_x4_out)

        conv3a_x8_out = self.conv3a_x8(torch.cat((pool3_Y_out, prelu3_x4_out), 1))
        prelu3a_x8_out = self.prelu3a_x8(conv3a_x8_out)
        conv3b_x8_out = self.conv3b_x8(prelu3a_x8_out)
        prelu3b_x8_out = self.prelu3b_x8(conv3b_x8_out)
        deconv4_x8_out = self.deconv4_x8(prelu3b_x8_out)
        prelu4_x8_out = self.prelu4_x8(deconv4_x8_out)

        conv4a_out = self.conv4a(torch.cat((prelu2_Y_out, prelu4_x8_out), 1))
        prelu4a_out = self.prelu4a(conv4a_out)
        conv4b_out = self.conv4b(prelu4a_out)
        prelu4b_out = self.prelu4b(conv4b_out)
        conv4c_out = self.conv4c(prelu4b_out)
        prelu4c_out = self.prelu4c(conv4c_out)
        hf_depth_out = self.conv5(prelu4c_out)

        depth_out = hf_depth_out + ups_lp_depth[:, :, 0:-1, 0:-1]

        for i in range(depth_out.size()[0]):
            depth_out[i, :, :, :] = depth_out[i, :, :, :] * (max_depth[i] - min_depth[i]) + min_depth[i]

        return hf_depth_out, depth_out


class Upsample8xDenoiseModel(nn.Module):
    def __init__(self):
        super(Upsample8xDenoiseModel, self).__init__()
        self.upsampler = Upsampler8x()

        self.ds1 = nn.Sequential(
            nn.Conv3d(1, 1, 7, stride=2, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds2 = nn.Sequential(
            nn.Conv3d(1, 1, 5, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds3 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(36, 36, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(36),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(28, 28, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(28),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(41, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.regress = nn.Sequential(
            nn.Conv3d(16, 1, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
        )

        self.conv0 = nn.Sequential(
            nn.Conv3d(1, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.intensity_ds = nn.Sequential(
            torch.nn.Conv2d(1, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        self.refine_depth1 = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.refine_depth2 = nn.Sequential(
            torch.nn.Conv2d(33, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids_in = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids1 = _DS_Block()
        self.ids2 = _DS_Block()

        self.iskip = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, (6, 6), (2, 2), 2)
        )

        self.iup1 = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup1_refine = nn.Sequential(
            torch.nn.Conv2d(64, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup2 = nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup2_refine = nn.Sequential(
            torch.nn.Conv2d(48, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup3 = nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
        )
        self.iup3_refine = nn.Sequential(
            torch.nn.Conv2d(40, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        # https://github.com/twtygqyy/pytorch-LapSRN/blob/master/lapsrn.py#L6
        def get_upsample_filter(size):
            """Make a 2D bilinear kernel suitable for upsampling"""
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1.
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size, :size]
            filter = (1 - abs(og[0] - center) / factor) * \
                     (1 - abs(og[1] - center) / factor) * \
                     (1 - abs(og[2] - center) / factor)
            return torch.from_numpy(filter).float()

        for n in [self.up1, self.up2, self.up3, self.ds1, self.ds2, self.ds3]:
            for m in n:
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                    c1, c2, d, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, d, h, w).repeat(c1, c2, 1, 1, 1)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, spad, intensity):

        # downsample intensity image
        intensity_ds_out = self.intensity_ds(intensity)
        tiled_intensity_ds_out = intensity_ds_out.repeat(1, 1024, 1, 1).unsqueeze(1)
        # pass spad through autoencoder
        smax = torch.nn.Softmax2d()

        ds1_out = self.ds1(spad)
        ds2_out = self.ds2(ds1_out)
        ds3_out = self.ds3(ds2_out)

        conv0_out = self.conv0(spad)
        conv1_out = self.conv1(ds1_out)
        conv2_out = self.conv2(ds2_out)
        conv3_out = self.conv3(ds3_out)

        up3_out = self.up3(conv3_out)
        up2_out = self.up2(torch.cat((conv2_out, up3_out), 1))
        up1_out = self.up1(torch.cat((conv1_out, up2_out), 1))
        up0_out = torch.cat((conv0_out, up1_out), 1)

        refine_out = self.refine(torch.cat((tiled_intensity_ds_out, up0_out), 1))
        regress_out = self.regress(refine_out)

        # squeeze and softmax for pixelwise classification loss
        denoise_out = torch.squeeze(regress_out, 1)
        if self.training:
            smax_denoise_out = smax(denoise_out)
        else:
            # make the softmax output more peaky
            smax_denoise_out = smax(1e6*denoise_out)

        # soft argmax
        weights = Variable(torch.linspace(0, 1, steps=1024).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax_denoise_out
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        # upsampling network
        hf_depth_out, depth_out = self.upsampler(soft_argmax, intensity)

        return denoise_out, soft_argmax, hf_depth_out, depth_out


class Upsampler2x(nn.Module):
    def __init__(self):
        super(Upsampler2x, self).__init__()

        # intensity
        self.conv1_Y = nn.Sequential(
            nn.Conv2d(1, 49, 7, 1, 3, bias=True)
        )
        self.prelu1_Y = nn.Sequential(
            nn.PReLU(49)
        )
        self.conv2_Y = nn.Sequential(
            nn.Conv2d(49, 32, 5, 1, 2, bias=True)
        )
        self.prelu2_Y = nn.Sequential(
            nn.PReLU(32)
        )

        # depth
        self.conv1_D = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 2, bias=True)
        )
        self.prelu1_D = nn.Sequential(
            nn.PReLU(64)
        )
        self.deconv2_D = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, 2, 2, bias=True)
        )
        self.prelu2_D = nn.Sequential(
            nn.PReLU(32)
        )

        # <--- concatenation pool4_Y and deconv2_D --->

        self.conv2a = nn.Sequential(
            nn.Conv2d(64, 32, 5, 1, 2, bias=True)
        )
        self.prelu2a = nn.Sequential(
            nn.PReLU(32)
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2, bias=True)
        )
        self.prelu2b = nn.Sequential(
            nn.PReLU(32)
        )
        self.conv2c = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2, bias=True)
        )
        self.prelu2c = nn.Sequential(
            nn.PReLU(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 1, 5, 1, 2, bias=True)
        )
        self.lp_filter = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        )

        # initialize low pass filter
        for n in [self.lp_filter]:
            for m in n:
                if isinstance(m, nn.Conv2d):
                    c1, f, h, w = m.weight.data.size()
                    weight = torch.ones(3, 3) / 9
                    m.weight.data = weight.view(1, 1, h, w).repeat(c1, f, 1, 1)
                    if m.bias is not None:
                        m.bias.data.zero_()

        # set requires gradient to false
        for f in self.lp_filter.parameters():
            f.requires_grad = False

    def forward(self, depth, intensity):

        intensity = intensity[:, :, 0:-1, 0:-1]

        # prep intensity image
        lp_intensity = self.lp_filter(intensity)
        hp_intensity = intensity - lp_intensity
        for i in range(hp_intensity.size()[0]):
            hp_intensity[i, :, :, :] = (hp_intensity[i, :, :, :] - torch.min(hp_intensity[i, :, :, :])) / \
                (torch.max(hp_intensity[i, :, :, :]) - torch.min(hp_intensity[i, :, :, :]))

        # prep depth image
        min_depth = Variable(torch.zeros(depth.size()[0]).type(dtype))
        max_depth = Variable(torch.zeros(depth.size()[0]).type(dtype))
        depth_normalized = Variable(torch.zeros(depth.size()).type(dtype))
        for i in range(depth.size()[0]):
            min_depth[i] = torch.min(depth[i, :, :, :])
            max_depth[i] = torch.max(depth[i, :, :, :])
            depth_normalized[i, :, :, :] = (depth[i, :, :, :] - min_depth[i]) / (max_depth[i] - min_depth[i])

        lp_depth = self.lp_filter(depth_normalized)
        hp_depth = depth_normalized - lp_depth

        # get high-resolution lp_depth
        lp_depth_np = lp_depth.data.cpu().numpy()
        ups_lp_depth_np = np.zeros((lp_depth_np.shape[0], lp_depth_np.shape[2]*2, lp_depth_np.shape[3]*2))
        for i in range(lp_depth_np.shape[0]):
            ups_lp_depth_np[i, :, :] = skimage.transform.rescale(np.squeeze(lp_depth_np[i, :, :, :]), 2, order=3, mode='symmetric', clip=False)

        ups_lp_depth = Variable(torch.from_numpy(ups_lp_depth_np).unsqueeze(1).type(dtype))

        # pass through intensity network
        conv1_Y_out = self.conv1_Y(hp_intensity)
        prelu1_Y_out = self.prelu1_Y(conv1_Y_out)
        conv2_Y_out = self.conv2_Y(prelu1_Y_out)
        prelu2_Y_out = self.prelu2_Y(conv2_Y_out)

        # depth network
        conv1_D_out = self.conv1_D(hp_depth)
        prelu1_D_out = self.prelu1_D(conv1_D_out)
        deconv2_D_out = self.deconv2_D(prelu1_D_out)
        prelu2_D_out = self.prelu2_D(deconv2_D_out)

        conv2a_out = self.conv2a(torch.cat((prelu2_Y_out, prelu2_D_out), 1))
        prelu2a_out = self.prelu2a(conv2a_out)
        conv2b_out = self.conv2b(prelu2a_out)
        prelu2b_out = self.prelu2b(conv2b_out)
        conv2c_out = self.conv2c(prelu2b_out)
        prelu2c_out = self.prelu2c(conv2c_out)
        hf_depth_out = self.conv3(prelu2c_out)

        depth_out = hf_depth_out + ups_lp_depth[:, :, 0:-1, 0:-1]

        for i in range(depth_out.size()[0]):
            depth_out[i, :, :, :] = depth_out[i, :, :, :] * (max_depth[i] - min_depth[i]) + min_depth[i]
        return hf_depth_out, depth_out


class Upsample2xDenoiseModel(nn.Module):
    def __init__(self):
        super(Upsample2xDenoiseModel, self).__init__()
        self.epochs = 0
        self.iters = 0
        self.upsampler = Upsampler2x()

        self.ds1 = nn.Sequential(
            nn.Conv3d(1, 1, 7, stride=2, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds2 = nn.Sequential(
            nn.Conv3d(1, 1, 5, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
        self.ds3 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(36, 36, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(36),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(28, 28, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(28),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 6, stride=2, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.refine = nn.Sequential(
            nn.Conv3d(41, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.regress = nn.Sequential(
            nn.Conv3d(16, 1, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
        )

        self.conv0 = nn.Sequential(
            nn.Conv3d(1, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 4, 9, stride=1, padding=4, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 7, stride=1, padding=3, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
            nn.Conv3d(12, 12, 5, stride=1, padding=2, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(12),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )

        self.intensity_ds = nn.Sequential(
            torch.nn.Conv2d(1, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (7, 7), (2, 2), 3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        self.refine_depth1 = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.refine_depth2 = nn.Sequential(
            torch.nn.Conv2d(33, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids_in = nn.Sequential(
            torch.nn.Conv2d(1, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        self.ids1 = _DS_Block()
        self.ids2 = _DS_Block()

        self.iskip = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, (6, 6), (2, 2), 2)
        )

        self.iup1 = nn.Sequential(
            torch.nn.ConvTranspose2d(65, 32, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup1_refine = nn.Sequential(
            torch.nn.Conv2d(64, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.iup2 = nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup2_refine = nn.Sequential(
            torch.nn.Conv2d(48, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        self.iup3 = nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, (6, 6), (2, 2), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
        )
        self.iup3_refine = nn.Sequential(
            torch.nn.Conv2d(40, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), 2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (5, 5), (1, 1), 2),
        )

        # https://github.com/twtygqyy/pytorch-LapSRN/blob/master/lapsrn.py#L6
        def get_upsample_filter(size):
            """Make a 2D bilinear kernel suitable for upsampling"""
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1.
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size, :size]
            filter = (1 - abs(og[0] - center) / factor) * \
                     (1 - abs(og[1] - center) / factor) * \
                     (1 - abs(og[2] - center) / factor)
            return torch.from_numpy(filter).float()

        for n in [self.up1, self.up2, self.up3, self.ds1, self.ds2, self.ds3]:
            for m in n:
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                    c1, c2, d, h, w = m.weight.data.size()
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, d, h, w).repeat(c1, c2, 1, 1, 1)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, spad, intensity):

        # downsample intensity image
        #  intensity_ds_out = self.intensity_ds(intensity)
        # do 2x downsampling
        intensity_ds_out = torch.nn.AvgPool2d(2, stride=2)(intensity)

        tiled_intensity_ds_out = intensity_ds_out.repeat(1, spad.size()[2], 1, 1).unsqueeze(1)
        # pass spad through autoencoder
        smax = torch.nn.Softmax2d()

        ds1_out = self.ds1(spad)
        ds2_out = self.ds2(ds1_out)
        ds3_out = self.ds3(ds2_out)

        conv0_out = self.conv0(spad)
        conv1_out = self.conv1(ds1_out)
        conv2_out = self.conv2(ds2_out)
        conv3_out = self.conv3(ds3_out)

        up3_out = self.up3(conv3_out)
        up2_out = self.up2(torch.cat((conv2_out, up3_out), 1))
        up1_out = self.up1(torch.cat((conv1_out, up2_out), 1))
        up0_out = torch.cat((conv0_out, up1_out), 1)

#         refine_out = self.refine(up0_out)
        refine_out = self.refine(torch.cat((tiled_intensity_ds_out, up0_out), 1))
        regress_out = self.regress(refine_out)

        # squeeze and softmax for pixelwise classification loss
        denoise_out = torch.squeeze(regress_out, 1)
        if self.training:
            smax_denoise_out = smax(denoise_out)
        else:
            # make the softmax output more peaky
            smax_denoise_out = smax(1e6*denoise_out)


        # soft argmax
        weights = Variable(torch.linspace(0, 1, steps=spad.size()[2]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax_denoise_out
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        # upsampling network
        hf_depth_out, depth_out = self.upsampler(soft_argmax, intensity)

        return denoise_out, soft_argmax, hf_depth_out, depth_out

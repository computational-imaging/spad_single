# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    # __constants__ = ["scale_factor", "mode"]

    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        if self.mode in ["linear", "bilinear", "trilinear"]:
            self.interp = lambda x : F.interpolate(x, scale_factor=self.scale_factor,
                                                   mode=self.mode, align_corners=True)
        else:
            self.interp = lambda x : F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        
    def extra_repr(self):
        return "scale_factor={}, mode={}".format(self.scale_factor, self.mode)

    def forward(self, x):
        x = self.interp(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, upsampling='bilinear'):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if upsampling == 'bilinear' or upsampling == 'nearest':
            self.up = Upsample(scale_factor=2, mode=upsampling)
            # self.up = lambda input_: F.interpolate(input_, scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv(x1)
        return x1


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

def expand_and_cat(ex, cat_to):
    return torch.cat([ex.expand(-1, -1, cat_to.size()[2], cat_to.size()[3]), cat_to], 1)
    
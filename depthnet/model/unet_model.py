# full assembly of the sub-parts to form the complete net

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .unet_parts import (up, down, inconv, outconv, double_conv, \
                         expand_and_cat, Upsample, to_logprobs)

class UNet(nn.Module):
    def __init__(self, input_nc, output_nc, upsampling_mode="bilinear", upnorm=nn.BatchNorm2d, **kwargs):
        super(UNet, self).__init__()
        self.inc = inconv(input_nc, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, upsampling_mode, upnorm)
        self.up2 = up(512, 128, upsampling_mode, upnorm)
        self.up3 = up(256, 64, upsampling_mode, upnorm)
        self.up4 = up(128, 64, upsampling_mode, upnorm)
        self.outc = outconv(64, output_nc)

    def forward(self, input_):
        x = input_["rgb"]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class UNetWithHints(nn.Module):
    def __init__(self, input_nc, output_nc, hist_len, num_hints_layers, len_hints_layers,
                 upsampling_mode="bilinear", upnorm=nn.BatchNorm2d,
                 **kwargs):
        super(UNetWithHints, self).__init__()
        self.unet = UNet(input_nc, output_nc, upsampling_mode, upnorm)
        self.hist_len = hist_len
        self.num_hints_layers = num_hints_layers

        # Create hints network
        assert num_hints_layers > 0
        hints_output = len_hints_layers
        hints = OrderedDict([("hints_conv_0", nn.Conv2d(self.hist_len, hints_output, kernel_size=1))])
        hints.update({"hints_relu_1": nn.ReLU(True)})
        j = 2
        for _ in range(num_hints_layers-1):
            hints.update({"hints_conv_{}".format(j): nn.Conv2d(hints_output, hints_output, kernel_size=1)})
            j += 1
            hints.update({"hints_relu_{}".format(j): nn.ReLU(True)})
            j += 1

        self.unet.up1 = up(1024+hints_output, 256, upsampling_mode) # Concatenate the output of the global hints
        self.global_hints = nn.Sequential(hints)
        self.bottleneck_conv = double_conv(512+hints_output, 512+hints_output)

    def forward(self, input_):
        rgb = input_["rgb"]
        hist = input_["hist"]
        mask = input_["mask"] # For masking away unknown depth values

        x1 = self.unet.inc(rgb)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        x5 = self.unet.down4(x4)

        y = self.global_hints(hist)
        z = expand_and_cat(y, x5)
        z = self.bottleneck_conv(z)

        x = self.unet.up1(z, x4)
        x = self.unet.up2(x, x3)
        x = self.unet.up3(x, x2)
        x = self.unet.up4(x, x1)
        x = self.unet.outc(x)
        # x = F.relu(x, True) # Map depth to [0, inf)
        return x


class UNetMultiScaleHints(nn.Module):
    def __init__(self, input_nc, output_nc, hist_len, num_hints_layers, len_hints_layers,
                 upsampling_mode="bilinear",
                 **kwargs):
        super(UNetMultiScaleHints, self).__init__()
        self.unet = UNet(input_nc, output_nc, upsampling_mode)
        self.hist_len = hist_len
        self.num_hints_layers = num_hints_layers

        # Create hints network
        assert num_hints_layers > 0
        hints_output = len_hints_layers
        hints = OrderedDict([("hints_conv_0", nn.Conv2d(self.hist_len, hints_output, kernel_size=1))])
        hints.update({"hints_relu_1": nn.ReLU(True)})
        j = 2
        for _ in range(num_hints_layers-1):
            hints.update({"hints_conv_{}".format(j): nn.Conv2d(hints_output, hints_output, kernel_size=1)})
            j += 1
            hints.update({"hints_relu_{}".format(j): nn.ReLU(True)})
            j += 1
        self.global_hints = nn.Sequential(hints)


        self.unet.up1 = up(1024+hist_len, 256, upsampling_mode) # Concatenate the output of the global hints
        self.unet.up2 = up(512+hist_len, 128, upsampling_mode) # Concatenate the output of the global hints
        self.unet.up3 = up(256+hist_len, 64, upsampling_mode) # Concatenate the output of the global hints
        self.unet.up4 = up(128+hist_len, 64, upsampling_mode) # Concatenate the output of the global hints


    def forward(self, input_):
        rgb = input_["rgb"]
        hist = input_["hist"]
        mask = input_["mask"] # For masking away unknown depth values

        x1 = self.unet.inc(rgb)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        x5 = self.unet.down4(x4)

        y = self.global_hints(hist)
        x4 = expand_and_cat(y, x4)
        x3 = expand_and_cat(y, x3)
        x2 = expand_and_cat(y, x2)
        x1 = expand_and_cat(y, x1)

        x = self.unet.up1(x5, x4)
        x = self.unet.up2(x, x3)
        x = self.unet.up3(x, x2)
        x = self.unet.up4(x, x1)
        x = self.unet.outc(x)
        # x = F.relu(x, True) # Map depth to [0, inf)
        return x


class UNetDORN(nn.Module):
    def __init__(self, input_nc, sid_bins, upsampling_mode="bilinear", upnorm=nn.BatchNorm2d, **kwargs):
        super(UNetDORN, self).__init__()
        self.unet = UNet(input_nc=input_nc, output_nc=2*sid_bins, upsampling_mode=upsampling_mode, upnorm=upnorm)
        self.unet.outc = nn.Conv2d(64, 2*sid_bins, kernel_size=1, bias=False)
        self.sid_bins = sid_bins

    def forward(self, input_):
        x = self.unet(input_)

        if torch.isnan(x).any():
            print("x is nan")
        log_ord_c0, log_ord_c1 = to_logprobs(x)
        return log_ord_c0, log_ord_c1

class UNetDORNWithHints(nn.Module):
    def __init__(self, input_nc, sid_bins, hist_len, num_hints_layers, len_hints_layers,
                 upsampling_mode="bilinear", upnorm=nn.BatchNorm2d, **kwargs):
        super(UNetDORNWithHints, self).__init__()
        self.unet = UNet(input_nc, 2*sid_bins, upsampling_mode, upnorm)
        self.unet.outc = nn.Conv2d(64, 2*sid_bins, kernel_size=1, bias=False)
        self.sid_bins = sid_bins

        self.hist_len = hist_len
        self.num_hints_layers = num_hints_layers

        # Create hints network
        assert num_hints_layers > 0
        hints_output = len_hints_layers
        hints = OrderedDict([("hints_conv_0", nn.Conv2d(self.hist_len, hints_output, kernel_size=1))])
        hints.update({"hints_relu_1": nn.ReLU(True)})
        j = 2
        for _ in range(num_hints_layers-1):
            hints.update({"hints_conv_{}".format(j): nn.Conv2d(hints_output, hints_output, kernel_size=1)})
            j += 1
            hints.update({"hints_relu_{}".format(j): nn.ReLU(True)})
            j += 1

        self.unet.up1 = up(1024+hints_output, 256, upsampling_mode) # Concatenate the output of the global hints
        self.global_hints = nn.Sequential(hints)
        self.bottleneck_conv = double_conv(512+hints_output, 512+hints_output)

    def forward(self, input_):
        rgb = input_["rgb"]
        hist = input_["hist"]

        x1 = self.unet.inc(rgb)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        x5 = self.unet.down4(x4)

        y = self.global_hints(hist)
        z = expand_and_cat(y, x5)
        z = self.bottleneck_conv(z)

        x = self.unet.up1(z, x4)
        x = self.unet.up2(x, x3)
        x = self.unet.up3(x, x2)
        x = self.unet.up4(x, x1)
        x = self.unet.outc(x)

        log_ord_c0, log_ord_c1 = to_logprobs(x)
        return log_ord_c0, log_ord_c1

if __name__ == '__main__':
    # model = UNetDORN(3, 4)
    model = UNetDORNWithHints(3, 4, 10, 4, 20)
    input_ = {"rgb": torch.ones(1, 3, 32, 32),
              "hist": torch.randn(1, 10, 1, 1)}
    output = model(input_)
    print(output)



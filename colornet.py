import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class Block(nn.Module):
     def __init__(self, feat_in, feat_out, blocks, stride, dilated, batchnorm):
        super(Block, self).__init__()
        self.ksize = 3
        layer_0 = []
        layers = []
        
        dilation = 1
        if dilated:
            dilation = 2
        padding = dilation * (self.ksize - 1) // 2

        if stride == 2:
            layer_0.append(nn.Conv2d(feat_in, feat_out, self.ksize, stride=2, padding=padding, dilation=dilation))
        elif stride == 1/2:
            layer_0.append(nn.ConvTranspose2d(feat_in, feat_out, 4, stride=2, padding=1, dilation=1))
        else:
            layer_0.append(nn.Conv2d(feat_in, feat_out, self.ksize, stride=1, padding=padding, dilation=dilation))
        layers.append(nn.ReLU())

        for i in range(blocks-1):
            layers.append(nn.Conv2d(feat_out, feat_out, self.ksize, stride=1, padding=padding, dilation=dilation))
            layers.append(nn.ReLU())

        if batchnorm:
            layers.append(nn.BatchNorm2d(feat_out))

        self.layer_0 = nn.Sequential(*layer_0)
        self.layers = nn.Sequential(*layers)

     def forward(self, feat_in, feat_skip=None):
        x = self.layer_0(feat_in)
        if feat_skip is not None:
            x = x + feat_skip
        x = self.layers(x)
        return x

class ConfidencePrediction(nn.Module):
    def __init__(self, feat_in):
        super(ConfidencePrediction, self).__init__()
        self.layer3 = nn.Conv2d(feat_in*4, 256, 3, 1, 1)
        self.layer4 = nn.ConvTranspose2d(feat_in*8, 256, 4, 2, 1)
        self.layer5 = nn.ConvTranspose2d(feat_in*8, 256, 4, 2, 1)
        self.layer6 = nn.ConvTranspose2d(feat_in*8, 256, 4, 2, 1)
        self.layer7 = nn.ConvTranspose2d(feat_in*8, 256, 4, 2, 1)
        self.layer8 = nn.Conv2d(feat_in*4, 256, 3, 1, 1)
        self.relu = nn.ReLU()
        self.squish = nn.Conv2d(256, 1, 1, 1)
        self.up1 = nn.ConvTranspose2d(1, 1, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(1, 1, 4, 2, 1)
    def forward(self, layer3, layer4, layer5, layer6, layer7, layer8):
        l3_out = self.layer3(layer3)
        l4_out = self.layer4(layer4)
        l5_out = self.layer5(layer5)
        l6_out = self.layer6(layer6)
        l7_out = self.layer7(layer7)
        l8_out = self.layer8(layer8)
        y = self.relu(l3_out + l4_out + l5_out + l6_out + l7_out + l8_out)
        y = self.squish(y)
        y = self.up1(y)
        y = self.up2(y)
        return y

class ColorNet(nn.Module):
    def __init__(self, confidence=False):
        super(ColorNet, self).__init__()
        self.feat_in = 64
        self.confidence = confidence
        self.confidence_network = ConfidencePrediction(self.feat_in)

        self.in1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.in2 = nn.Conv2d(2, 64, 3, 1, 1)
        self.in_relu = nn.ReLU()
        self.layer1 = self.make_layer(blocks=1, stride=1, expansion=1, dilated=False)
        self.layer1_short = nn.Conv2d(self.feat_in, 2*self.feat_in, 3, 1, 1)
        self.layer2 = self.make_layer(blocks=2, stride=2, expansion=2, dilated=False)
        self.layer3 = self.make_layer(blocks=3, stride=2, expansion=2, dilated=False)
        self.layer4 = self.make_layer(blocks=3, stride=2, expansion=2, dilated=False)
        self.layer5 = self.make_layer(blocks=3, stride=1, expansion=1, dilated=True)
        self.layer6 = self.make_layer(blocks=3, stride=1, expansion=1, dilated=True)
        self.layer7 = self.make_layer(blocks=3, stride=1, expansion=1, dilated=False)
        self.layer8 = self.make_layer(blocks=3, stride=1/2, expansion=1/2, dilated=False)
        self.layer9 = self.make_layer(blocks=2, stride=1/2, expansion=1/2, dilated=False)
        self.layer10 = self.make_layer(blocks=2, stride=1/2, expansion=1, dilated=False, batchnorm=False)
        self.out1 = nn.Conv2d(self.feat_in, 1, 3, 1, 1)

    def make_layer(self, blocks=2, stride=1, dilated=False, expansion=1, batchnorm=True):
        feat_out = int(expansion * self.feat_in)
        block = Block(self.feat_in, feat_out, blocks, stride, dilated, batchnorm)
        self.feat_in = feat_out
        return block

    def forward(self, intensity, guidance):
        x = self.in1(intensity)
        x = x + self.in2(guidance)
        x = self.in_relu(x)
        x = self.layer1(x)
        layer1_out = self.layer1_short(x)
        x = self.layer2(x)
        layer2_out = x.clone()
        x = self.layer3(x)
        layer3_out = x.clone()
        x = self.layer4(x)
        layer4_out = x.clone()
        x = self.layer5(x)
        layer5_out = x.clone()
        x = self.layer6(x)
        layer6_out = x.clone()
        x = self.layer7(x)
        layer7_out = x.clone()
        x = self.layer8(x, feat_skip=layer3_out)
        layer8_out = x.clone()
        x = self.layer9(x, feat_skip=layer2_out)
        x = self.layer10(x, feat_skip=layer1_out)
        x = self.out1(x)

        if self.confidence:
            conf = self.confidence_network(layer3_out, layer4_out, layer5_out, layer6_out, layer7_out, layer8_out)
            return x, conf
        else:
            return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torchvision.models.densenet import densenet169

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.convA = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.convB = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x, concat_with):
        x = self.upsample(x)
        x = torch.cat((x, concat_with), dim=1)
        x = self.convA(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.convB(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        return x
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name + '_upsampling2d')(tensor)
            up_i = Concatenate(name=name + '_concat')(
                [up_i, base_model.get_layer(concat_with).output])  # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape,
                         name='conv2')(base_model.output)

        decoder = upproject(decoder, int(decode_filters / 2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters / 4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters / 8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters / 16), 'up4', concat_with='conv1/relu')


class OrdNet(nn.Module):
    def __init__(self):
        super(OrdNet, self).__init__()
        # Load densenet and modify the forward function to match
        # keras.applications' densenet169 when include_top=False (as in the DenseDepth paper)
        self.encoder = densenet169(pretrained=True)

        self.pools = {}
        self.activation = {}
        def get_activation(name, activation):
            def hook(model, input, output):
                activation[name] = output.clone()
            return hook
        self.encoder.features.relu0.register_forward_hook(get_activation("conv0", self.activation))
        self.encoder.features.pool0.register_forward_hook(get_activation("pool0", self.activation))
        self.encoder.features.transition1.pool.register_forward_hook(get_activation("pool1", self.activation))
        self.encoder.features.transition2.pool.register_forward_hook(get_activation("pool2", self.activation))
        # self.encoder.features.transition3.pool.register_forward_hook(get_activation("pool3", self.activation))

        # Make Decoder
        decoder_channels, conv0_channels, pool0_channels, pool1_channels, pool2_channels  = \
            self.get_num_encoder_channels()
        self.conv2 = nn.Conv2d(in_channels=decoder_channels, out_channels=decoder_channels, kernel_size=1, stride=1, padding=0)
        self.up1 = UpBlock(decoder_channels + pool2_channels, decoder_channels//2)
        self.up2 = UpBlock(decoder_channels//2 + pool1_channels, decoder_channels//4)
        self.up3 = UpBlock(decoder_channels//4 + pool0_channels , decoder_channels//8)
        self.up4 = UpBlock(decoder_channels//8 + conv0_channels, decoder_channels//16)
        self.conv3 = nn.Conv2d(in_channels=decoder_channels//16, out_channels=1, kernel_size=3, padding=1, stride=1)
    #
    def get_num_encoder_channels(self):
        dummy_input = torch.randn((1, 3, 200, 200))
        dummy_output = self.encoder.features(dummy_input)
        # print(dummy_output.shape)
        print(self.activation)
        return dummy_output.shape[1], self.activation["conv0"].shape[1], \
               self.activation["pool0"].shape[1], self.activation["pool1"].shape[1], \
               self.activation["pool2"].shape[1]

    def forward(self, x):
        x = self.encoder.features(x)
        x = F.relu(x, inplace=True)
        # self.activation["pool0"]
        x = self.conv2(x)
        x = self.up1(x, self.activation["pool2"])
        x = self.up2(x, self.activation["pool1"])
        x = self.up3(x, self.activation["pool0"])
        x = self.up4(x, self.activation["conv0"])
        x = self.conv3(x)
        return x

if __name__ == "__main__":
    net = OrdNet()
    # print(net)
    input_= torch.randn((1, 3, 480, 640))
    output = net(input_)
    print(output.shape)

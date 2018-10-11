import torch
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict

def create_sublayer_dict(input_nc, output_nc, layer_index, nconvs, norm_layer, **conv_kwargs):
    sublayer=OrderedDict()
    j = 0
    sublayer.update({"conv{}_{}".format(layer_index, j): nn.Conv2d(input_nc, output_nc, **conv_kwargs)})
    j += 1
    sublayer.update({"relu{}_{}".format(layer_index, j): nn.ReLU(True)})
    for i in range(nconvs-1):
        j += 1
        sublayer.update({"conv{}_{}".format(layer_index, j): nn.Conv2d(output_nc, output_nc, **conv_kwargs)})
        j += 1
        sublayer.update({"relu{}_{}".format(layer_index, j): nn.ReLU(True)})
    if norm_layer:
        j += 1
        sublayer.update({"norm{}_{}".format(layer_index, j): norm_layer(output_nc)})
    return sublayer

class DepthNet(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(DepthNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        use_bias = True

        # Conv1
        model1 = create_sublayer_dict(input_nc, 32, 1, 2, norm_layer, 
                                      kernel_size=3, stride=1, padding=1, bias=use_bias)
        # add a subsampling operation (in self.forward())

        # Conv2
        model2 = create_sublayer_dict(32, 64, 2, 2, norm_layer,
                                      kernel_size=3, stride=1, padding=1, bias=use_bias)
        # add a subsampling layer operation (in self.forward())

        # Conv3
        model3 = create_sublayer_dict(64, 128, 3, 3, norm_layer,
                                      kernel_size=3, stride=1, padding=1, bias=use_bias)
        # add a subsampling layer operation

        # Conv4
        model4 = create_sublayer_dict(128, 256, 4, 3, norm_layer,
                                      kernel_size=3, stride=1, padding=1, bias=use_bias)

        # Conv5
        model5 = create_sublayer_dict(256, 256, 5, 3, norm_layer,
                                      kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias)

        # Conv6
        model6 = create_sublayer_dict(256, 256, 6, 3, norm_layer,
                                      kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias)

        # Conv7
        model7 = create_sublayer_dict(256, 256, 7, 3, norm_layer,
                                      kernel_size=3, stride=1, padding=1, bias=use_bias)

        # Conv7
        model8up=OrderedDict([("convt8_up", 
                               nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias))]
                            )
        model3short8=OrderedDict([("conv3short8",
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias))]
                                )

        model8=OrderedDict([("relu8_pre", nn.ReLU(True))])

        model8.update(create_sublayer_dict(128, 128, 8, 2, norm_layer,
                                      kernel_size=3, stride=1, padding=1, bias=use_bias))

        # Conv9
        model9up=OrderedDict([("convt9_up", 
                               nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=use_bias))]
                            )
        model2short9=OrderedDict([("conv2short9",
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias))]
                                )
        # add the two feature maps above        

        model9=OrderedDict([("relu9_pre", nn.ReLU(True))])

        model9.update(create_sublayer_dict(64, 64, 9, 1, norm_layer,
                                           kernel_size=3, stride=1, padding=1, bias=use_bias))

        # Conv10
        model10up=OrderedDict([("conv10_up", 
                               nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=use_bias))]
                             )
        model1short10=OrderedDict([("conv1short10", 
                                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=use_bias))]
                                 )
        # add the two feature maps above

        model10=OrderedDict([("relu10_pre", nn.ReLU(True))])
        model10.update({"conv10_0": nn.Conv2d(64, 64, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias)})
        model10.update({"leakyrelu10_1": nn.LeakyReLU(negative_slope=.2)})

        # Depth Map Regression Output
        model_out=OrderedDict([("conv_out",
                                nn.Conv2d(64, 1, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias))]
                             )
        model_out.update({"relu_out": nn.ReLU(True)}) # Depth should be in [0, +inf)

        self.model1 = nn.Sequential(model1)
        self.model2 = nn.Sequential(model2)
        self.model3 = nn.Sequential(model3)
        self.model4 = nn.Sequential(model4)
        self.model5 = nn.Sequential(model5)
        self.model6 = nn.Sequential(model6)
        self.model7 = nn.Sequential(model7)
        self.model8up = nn.Sequential(model8up)
        self.model8 = nn.Sequential(model8)
        self.model9up = nn.Sequential(model9up)
        self.model9 = nn.Sequential(model9)
        self.model10up = nn.Sequential(model10up)
        self.model10 = nn.Sequential(model10)
        self.model3short8 = nn.Sequential(model3short8)
        self.model2short9 = nn.Sequential(model2short9)
        self.model1short10 = nn.Sequential(model1short10)
        self.model_out = nn.Sequential(model_out)

    def forward(self, input_A):
        conv1_2 = self.model1(input_A)
        conv2_2 = self.model2(conv1_2[:,:,::2,::2]) # downsample
        conv3_3 = self.model3(conv2_2[:,:,::2,::2]) # downsample
        conv4_3 = self.model4(conv3_3[:,:,::2,::2]) # downsample
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3) # Shortcut
        conv8_3 = self.model8(conv8_up)
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2) # Shortcut
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2) # Shortcut
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        return out_reg


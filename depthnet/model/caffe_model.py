import caffe

import torch
import torch.nn as nn

class FrozenCaffeModule(nn.Module):
    def __init__(self, proto_file, caffemodel_file, device):
        """
        Load caffe model and weights
        :param proto_file: File path of caffe .prototxt file.
        :param weights_file: File path of corresponding caffe .caffemodel file
        """
        super(FrozenCaffeModule, self).__init__()

        self.proto_file = proto_file
        self.weights_file = caffemodel_file
        self.net = caffe.Net(proto_file, caffemodel_file, caffe.TEST)

        if device.type == "cuda":
            net.set_gpu_
    def forward(self, x):

        self.net.forward(x)
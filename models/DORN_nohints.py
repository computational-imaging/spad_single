import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.model_core import Model


class DORN_nyu_nohints(Model):
    """
    Deep Ordinal Regression Network

    Ported from caffe.

    Meant to be run as a part of a larger network.
    """
    def __init__(self, in_channels=3, in_height=257, in_width=353, sid_bins=136, frozen=True, pretrained=True,
                 state_dict_file="torch_params_nyuv2_first_flip.pth.tar", **kwargs):
        super(DORN_nyu, self).__init__()
        self.make_layers(in_channels, in_height, in_width, sid_bins)

        self.in_heignt = in_height
        self.in_width = in_width
        self.in_channels = in_channels
        self.sid_bins = sid_bins

        self.frozen = frozen
        self.pretrained = pretrained
        self.state_dict_file = state_dict_file
        if pretrained:
            self.load_state_dict(torch.load(state_dict_file))
        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def get_loss(self, input_, device):
        rgb = input_["rgb"].to(device)
        depth_pred = self.forward(rgb)
        prediction = self.to_logprobs(input_["rawdepth"].to(device))
        mask = input_["mask"].to(device)

        return ord

    @staticmethod
    def to_logprobs(x):
        """
        Compute the output log probabilities using the same method as in DORN_pytorch.
        :param x: The activations from the last layer: N x (2*sid_bins) x H x W
        :return:
            log_ord_c: Per pixel, a vector with the numbers log P(l > k)
            log_ord_c_comp: Per pixel, a vector with the numbers log (1 - P(l > k))

        By convention, x[:, ::2, :, :] corresponds to log P(l > k), and
        x[:, 1::2, :, :] corresponds to log (1 - P(l > k))
        """
        N = x.size(0)
        sid_bins = x.size(1) // 2
        H, W = x.size()[-2:]
        c_comp = x[:, ::2, :, :].clone()
        c = x[:, 1::2, :, :].clone()
        c_comp = c_comp.view(N, 1, -1)
        c = c.view(N, 1, -1)

        c_c_comp = torch.cat((c, c_comp), dim=1)
        log_ord = F.log_softmax(c_c_comp, dim=1)
        log_ord_c = log_ord[:, 0, :].clone()
        log_ord_c = log_ord_c.view(-1, sid_bins, H, W)
        log_ord_c_comp = log_ord[:, 1, :].clone()
        log_ord_c_comp = log_ord_c_comp.view(-1, sid_bins, H, W)
        return log_ord_c, log_ord_c_comp

    @staticmethod
    def ord_reg_loss(prediction, target, mask, size_average=True, eps=1e-6):
        """Calculates the Ordinal Regression loss
        :param prediction: a tuple (log_ord_c, log_ord_c_comp).
            log_ord_c is is an N x K x H x W tensor
            where each pixel location is a length K vector containing log-probabilities log P(l > 0),..., log P(l > K-1).

            The log_ord_c_comp is the same, but contains the log-probabilities log (1 - P(l > 0)),..., log (1 - P(l > K-1))
            instead.
        :param target - per-pixel vector of 0's and 1's such that if the true depth
        bin is k then the vector contains 1's up to entry k-1 and 0's for the remaining entries.
        e.g. if k = 3 and the total number of bins is 7 then

        target[:, i, j] = [1, 1, 1, 0, 0, 0, 0]

        :param mask - same size as prediction and target, 1.0 if that position is
        to be used in the loss calculation, 0 otherwise.
        :param size_average - whether or not to take the average over all the mask pixels.
        """
        log_ord_c, log_ord_c_comp = prediction
        # nbins = log_ord_c.size(1)
        mask_L = ((target > 0) & (mask > 0))
        mask_U = (((1. - target) > 0) & (mask > 0))

        out = -(torch.sum(log_ord_c[mask_L]) + torch.sum(log_ord_c_comp[mask_U]))
        if size_average:
            total = torch.sum(mask).item()
            if total > 0:
                return (1. / torch.sum(mask)) * out
            else:
                return torch.zeros(1)
        return out

    @staticmethod
    def ord_decode(prediction):
        pass

    def write_updates(self, writer, input_, output_, loss, it, tag):
        pass

    def write_eval(self, data, output_file, device):
        pass

    def evaluate_dir(self, output_dir, device):
        pass


    def make_layers(self, in_channels, in_height, in_width, sid_bins):
        """
        :param in_channels:
        :param in_height:
        :param in_width:
        :param sid_bins:
        :return:
        """
        # Resnet
        ### conv1
        self.conv1_1_3x3_s2 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1_1_3x3_s2_bn = nn.BatchNorm2d(64, momentum=0.95)
        self.conv1_1_3x3_s2_relu = nn.ReLU(inplace=True)

        self.conv1_2_3x3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_2_3x3_bn = nn.BatchNorm2d(64, momentum=0.95)
        self.conv1_2_3x3_relu = nn.ReLU(inplace=True)

        self.conv1_3_3x3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_3_3x3_bn = nn.BatchNorm2d(128, momentum=0.95)
        self.conv1_3_3x3_relu = nn.ReLU(inplace=True)

        self.pool1_3x3_s2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ### conv2_1 (reduce)
        self.conv2_1_1x1_reduce = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.conv2_1_1x1_reduce_bn = nn.BatchNorm2d(64, momentum=0.95)
        self.conv2_1_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv2_1_3x3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_1_3x3_bn = nn.BatchNorm2d(64, momentum=0.95)
        self.conv2_1_3x3_relu = nn.ReLU(inplace=True)

        self.conv2_1_1x1_increase = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.conv2_1_1x1_increase_bn = nn.BatchNorm2d(256, momentum=0.95)

        # proj skip
        self.conv2_1_1x1_proj = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        self.conv2_1_1x1_proj_bn = nn.BatchNorm2d(256, momentum=0.95)

        self.conv2_1_relu = nn.ReLU(inplace=True)

        ### conv2_2
        self.conv2_2_1x1_reduce = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.conv2_2_1x1_reduce_bn = nn.BatchNorm2d(64, momentum=0.95)
        self.conv2_2_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv2_2_3x3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2_3x3_bn = nn.BatchNorm2d(64, momentum=0.95)
        self.conv2_2_3x3_relu = nn.ReLU(inplace=True)

        self.conv2_2_1x1_increase = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.conv2_2_1x1_increase_bn = nn.BatchNorm2d(256, momentum=0.95)

        self.conv2_2_relu = nn.ReLU(inplace=True)

        ### conv2 3
        self.conv2_3_1x1_reduce = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.conv2_3_1x1_reduce_bn = nn.BatchNorm2d(64, momentum=0.95)
        self.conv2_3_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv2_3_3x3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_3_3x3_bn = nn.BatchNorm2d(64, momentum=0.95)
        self.conv2_3_3x3_relu = nn.ReLU(inplace=True)

        self.conv2_3_1x1_increase = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.conv2_3_1x1_increase_bn = nn.BatchNorm2d(256, momentum=0.95)

        self.conv2_3_relu = nn.ReLU(inplace=True)

        ### conv3_1 (reduce)
        self.conv3_1_1x1_reduce = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.conv3_1_1x1_reduce_bn = nn.BatchNorm2d(128, momentum=0.95)
        self.conv3_1_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv3_1_3x3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3_1_3x3_bn = nn.BatchNorm2d(128, momentum=0.95)
        self.conv3_1_3x3_relu = nn.ReLU(inplace=True)

        self.conv3_1_1x1_increase = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.conv3_1_1x1_increase_bn = nn.BatchNorm2d(512, momentum=0.95)

        # proj skip
        self.conv3_1_1x1_proj = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.conv3_1_1x1_proj_bn = nn.BatchNorm2d(512, momentum=0.95)

        self.conv3_1_relu = nn.ReLU(inplace=True)

        ### conv3_2
        self.conv3_2_1x1_reduce = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.conv3_2_1x1_reduce_bn = nn.BatchNorm2d(128, momentum=0.95)
        self.conv3_2_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv3_2_3x3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_2_3x3_bn = nn.BatchNorm2d(128, momentum=0.95)
        self.conv3_2_3x3_relu = nn.ReLU(inplace=True)

        self.conv3_2_1x1_increase = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.conv3_2_1x1_increase_bn = nn.BatchNorm2d(512, momentum=0.95)

        self.conv3_2_relu = nn.ReLU(inplace=True)

        ### conv3_3
        self.conv3_3_1x1_reduce = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.conv3_3_1x1_reduce_bn = nn.BatchNorm2d(128, momentum=0.95)
        self.conv3_3_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv3_3_3x3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_3_3x3_bn = nn.BatchNorm2d(128, momentum=0.95)
        self.conv3_3_3x3_relu = nn.ReLU(inplace=True)

        self.conv3_3_1x1_increase = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.conv3_3_1x1_increase_bn = nn.BatchNorm2d(512, momentum=0.95)

        self.conv3_3_relu = nn.ReLU(inplace=True)

        ### conv3_4
        self.conv3_4_1x1_reduce = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.conv3_4_1x1_reduce_bn = nn.BatchNorm2d(128, momentum=0.95)
        self.conv3_4_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv3_4_3x3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_4_3x3_bn = nn.BatchNorm2d(128, momentum=0.95)
        self.conv3_4_3x3_relu = nn.ReLU(inplace=True)

        self.conv3_4_1x1_increase = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.conv3_4_1x1_increase_bn = nn.BatchNorm2d(512, momentum=0.95)

        self.conv3_4_relu = nn.ReLU(inplace=True)

        ### conv4_1 (reduce)
        self.conv4_1_1x1_reduce = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.conv4_1_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_1_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_1_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_1_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_1_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_1_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_1_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        # proj skip
        self.conv4_1_1x1_proj = nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_1_1x1_proj_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_1_relu = nn.ReLU(inplace=True)

        ### conv4_2
        self.conv4_2_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_2_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_2_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_2_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_2_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_2_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_2_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_2_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_2_relu = nn.ReLU(inplace=True)

        ### conv4_3
        self.conv4_3_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_3_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_3_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_3_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_3_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_3_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_3_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_3_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_3_relu = nn.ReLU(inplace=True)

        ### conv4_4
        self.conv4_4_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_4_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_4_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_4_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_4_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_4_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_4_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_4_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_4_relu = nn.ReLU(inplace=True)

        ### conv4_5
        self.conv4_5_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_5_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_5_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_5_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_5_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_5_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_5_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_5_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_5_relu = nn.ReLU(inplace=True)

        ### conv4_6
        self.conv4_6_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_6_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_6_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_6_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_6_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_6_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_6_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_6_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_6_relu = nn.ReLU(inplace=True)

        ### conv4_7
        self.conv4_7_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_7_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_7_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_7_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_7_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_7_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_7_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_7_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_7_relu = nn.ReLU(inplace=True)

        ### conv4_8
        self.conv4_8_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_8_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_8_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_8_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_8_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_8_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_8_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_8_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_8_relu = nn.ReLU(inplace=True)

        ### conv4_9
        self.conv4_9_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_9_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_9_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_9_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_9_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_9_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_9_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_9_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_9_relu = nn.ReLU(inplace=True)

        ### conv4_10
        self.conv4_10_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_10_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_10_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_10_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_10_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_10_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_10_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_10_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_10_relu = nn.ReLU(inplace=True)

        ### conv4_11
        self.conv4_11_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_11_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_11_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_11_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_11_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_11_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_11_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_11_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_11_relu = nn.ReLU(inplace=True)

        ### conv4_12
        self.conv4_12_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_12_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_12_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_12_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_12_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_12_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_12_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_12_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_12_relu = nn.ReLU(inplace=True)

        ### conv4_13
        self.conv4_13_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_13_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_13_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_13_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_13_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_13_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_13_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_13_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_13_relu = nn.ReLU(inplace=True)

        ### conv4_14
        self.conv4_14_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_14_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_14_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_14_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_14_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_14_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_14_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_14_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_14_relu = nn.ReLU(inplace=True)

        ### conv4_15
        self.conv4_15_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_15_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_15_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_15_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_15_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_15_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_15_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_15_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_15_relu = nn.ReLU(inplace=True)

        ### conv4_16
        self.conv4_16_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_16_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_16_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_16_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_16_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_16_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_16_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_16_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_16_relu = nn.ReLU(inplace=True)

        ### conv4_17
        self.conv4_17_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_17_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_17_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_17_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_17_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_17_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_17_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_17_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_17_relu = nn.ReLU(inplace=True)

        ### conv4_18
        self.conv4_18_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_18_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_18_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_18_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_18_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_18_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_18_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_18_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_18_relu = nn.ReLU(inplace=True)

        ### conv4_19
        self.conv4_19_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_19_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_19_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_19_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_19_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_19_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_19_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_19_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_19_relu = nn.ReLU(inplace=True)

        ### conv4_20
        self.conv4_20_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_20_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_20_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_20_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_20_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_20_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_20_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_20_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_20_relu = nn.ReLU(inplace=True)

        ### conv4_21
        self.conv4_21_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_21_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_21_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_21_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_21_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_21_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_21_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_21_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_21_relu = nn.ReLU(inplace=True)

        ### conv4_22
        self.conv4_22_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_22_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_22_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_22_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_22_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_22_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_22_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_22_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_22_relu = nn.ReLU(inplace=True)

        ### conv4_23
        self.conv4_23_1x1_reduce = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_23_1x1_reduce_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_23_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv4_23_3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv4_23_3x3_bn = nn.BatchNorm2d(256, momentum=0.95)
        self.conv4_23_3x3_relu = nn.ReLU(inplace=True)

        self.conv4_23_1x1_increase = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.conv4_23_1x1_increase_bn = nn.BatchNorm2d(1024, momentum=0.95)

        self.conv4_23_relu = nn.ReLU(inplace=True)

        ### conv5_1 (reduce)
        self.conv5_1_1x1_reduce = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.conv5_1_1x1_reduce_bn = nn.BatchNorm2d(512, momentum=0.95)
        self.conv5_1_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv5_1_3x3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        self.conv5_1_3x3_bn = nn.BatchNorm2d(512, momentum=0.95)
        self.conv5_1_3x3_relu = nn.ReLU(inplace=True)

        self.conv5_1_1x1_increase = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.conv5_1_1x1_increase_bn = nn.BatchNorm2d(2048, momentum=0.95)

        # proj skip
        self.conv5_1_1x1_proj = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False)
        self.conv5_1_1x1_proj_bn = nn.BatchNorm2d(2048, momentum=0.95)

        self.conv5_1_relu = nn.ReLU(inplace=True)

        ### conv5_2
        self.conv5_2_1x1_reduce = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.conv5_2_1x1_reduce_bn = nn.BatchNorm2d(512, momentum=0.95)
        self.conv5_2_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv5_2_3x3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        self.conv5_2_3x3_bn = nn.BatchNorm2d(512, momentum=0.95)
        self.conv5_2_3x3_relu = nn.ReLU(inplace=True)

        self.conv5_2_1x1_increase = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.conv5_2_1x1_increase_bn = nn.BatchNorm2d(2048, momentum=0.95)

        self.conv5_2_relu = nn.ReLU(inplace=True)

        ### conv5_3
        self.conv5_3_1x1_reduce = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.conv5_3_1x1_reduce_bn = nn.BatchNorm2d(512, momentum=0.95)
        self.conv5_3_1x1_reduce_relu = nn.ReLU(inplace=True)

        self.conv5_3_3x3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        self.conv5_3_3x3_bn = nn.BatchNorm2d(512, momentum=0.95)
        self.conv5_3_3x3_relu = nn.ReLU(inplace=True)

        self.conv5_3_1x1_increase = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.conv5_3_1x1_increase_bn = nn.BatchNorm2d(2048, momentum=0.95)

        self.conv5_3_relu = nn.ReLU(inplace=True)
        # End ResNet

        # ASPP
        # Full Image Encoder
        # ceil_mode=True necessary to align with caffe behavior
        self.reduce_pooling = nn.AvgPool2d(kernel_size=8, stride=8, ceil_mode=True)
        self.drop_reduce = nn.Dropout2d(p=0.5, inplace=True)

        # Determine FC units:
        n_fc = self.forward_to_fc(torch.zeros(1, in_channels, in_height, in_width))

        self.ip1_depth = nn.Linear(n_fc, 512)
        self.relu_ip1 = nn.ReLU(inplace=True)
        # self.reshape_ip1 = # Just do the reshape in the forward pass
        self.conv6_1_soft = nn.Conv2d(512, 512, kernel_size=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        # self.interp_conv6_1 = # Do the expansion in the forward pass, to size H x W = 33 x 45
        # End Full Image Encoder

        # ASPP 1x1 conv
        self.aspp_1_soft = nn.Conv2d(2048, 512, kernel_size=1)
        self.relu_aspp_1 = nn.ReLU(inplace=True)
        self.conv6_2_soft = nn.Conv2d(512, 512, kernel_size=1)
        self.relu6_2 = nn.ReLU(inplace=True)
        # End ASPP 1x1 conv

        # ASPP dilation 4
        self.aspp_2_soft = nn.Conv2d(2048, 512, kernel_size=3, padding=4, dilation=4)
        self.relu_aspp_2 = nn.ReLU(inplace=True)
        self.conv6_3_soft = nn.Conv2d(512, 512, kernel_size=1)
        self.relu6_3 = nn.ReLU(inplace=True)
        # End ASPP dilation 4

        # ASPP dilation 8
        self.aspp_3_soft = nn.Conv2d(2048, 512, kernel_size=3, padding=8, dilation=8)
        self.relu_aspp_3 = nn.ReLU(inplace=True)
        self.conv6_4_soft = nn.Conv2d(512, 512, kernel_size=1)
        self.relu6_4 = nn.ReLU(inplace=True)
        # End ASPP dilation 8

        # ASPP dilation 12
        self.aspp_4_soft = nn.Conv2d(2048, 512, kernel_size=3, padding=12, dilation=12)
        self.relu_aspp_4 = nn.ReLU(inplace=True)
        self.conv6_5_soft = nn.Conv2d(512, 512, kernel_size=1)
        self.relu6_5 = nn.ReLU(inplace=True)
        # End ASPP dilation 12

        # Concatenate

        self.drop_conv6 = nn.Dropout2d(p=0.5, inplace=True)
        self.conv7_soft = nn.Conv2d(512*5, 2048, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop_conv7 = nn.Dropout2d(p=0.5, inplace=True)

        self.conv8 = nn.Conv2d(2048, sid_bins, kernel_size=1)

    def forward_to_fc(self, x):
        """
        Calculate the number of inputs to the fully connected layer by propagating forward.
        :param x: input with expected usual size
        :return: The number of units in the fully-connected layer
        """
        with torch.no_grad():
            # Resnet
            ### conv1
            x = self.conv1_1_3x3_s2(x)
            # self.first_conv = x.clone()
            x = self.conv1_1_3x3_s2_bn(x)
            x = self.conv1_1_3x3_s2_relu(x)

            x = self.conv1_2_3x3(x)
            x = self.conv1_2_3x3_bn(x)
            x = self.conv1_2_3x3_relu(x)

            x = self.conv1_3_3x3(x)
            x = self.conv1_3_3x3_bn(x)
            x = self.conv1_3_3x3_relu(x)

            x = self.pool1_3x3_s2(x)
            # self.conv1_out = x.clone()
            ### conv2_1 (reduce)
            x1 = self.conv2_1_1x1_reduce(x)
            x1 = self.conv2_1_1x1_reduce_bn(x1)
            x1 = self.conv2_1_1x1_reduce_relu(x1)

            x1 = self.conv2_1_3x3(x1)
            x1 = self.conv2_1_3x3_bn(x1)
            x1 = self.conv2_1_3x3_relu(x1)

            x1 = self.conv2_1_1x1_increase(x1)
            x1 = self.conv2_1_1x1_increase_bn(x1)

            # proj skip
            x2 = self.conv2_1_1x1_proj(x)
            x2 = self.conv2_1_1x1_proj_bn(x2)

            x = x1 + x2
            x = self.conv2_1_relu(x)
            # print("conv2", x.size())
            ### conv2_2
            x1 = self.conv2_2_1x1_reduce(x)
            x1 = self.conv2_2_1x1_reduce_bn(x1)
            x1 = self.conv2_2_1x1_reduce_relu(x1)

            x1 = self.conv2_2_3x3(x1)
            x1 = self.conv2_2_3x3_bn(x1)
            x1 = self.conv2_2_3x3_relu(x1)

            x1 = self.conv2_2_1x1_increase(x1)
            x1 = self.conv2_2_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv2_2_relu(x)

            ### conv2 3
            x1 = self.conv2_3_1x1_reduce(x)
            x1 = self.conv2_3_1x1_reduce_bn(x1)
            x1 = self.conv2_3_1x1_reduce_relu(x1)

            x1 = self.conv2_3_3x3(x1)
            x1 = self.conv2_3_3x3_bn(x1)
            x1 = self.conv2_3_3x3_relu(x1)

            x1 = self.conv2_3_1x1_increase(x1)
            x1 = self.conv2_3_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv2_3_relu(x)
            # self.conv2_out = x.clone()

            ### conv3_1 (reduce)
            x1 = self.conv3_1_1x1_reduce(x)
            x1 = self.conv3_1_1x1_reduce_bn(x1)
            x1 = self.conv3_1_1x1_reduce_relu(x1)

            x1 = self.conv3_1_3x3(x1)
            x1 = self.conv3_1_3x3_bn(x1)
            x1 = self.conv3_1_3x3_relu(x1)

            x1 = self.conv3_1_1x1_increase(x1)
            x1 = self.conv3_1_1x1_increase_bn(x1)

            # proj skip
            x2 = self.conv3_1_1x1_proj(x)
            x2 = self.conv3_1_1x1_proj_bn(x2)

            x = x1 + x2
            x = self.conv3_1_relu(x)

            ### conv3_2
            x1 = self.conv3_2_1x1_reduce(x)
            x1 = self.conv3_2_1x1_reduce_bn(x1)
            x1 = self.conv3_2_1x1_reduce_relu(x1)

            x1 = self.conv3_2_3x3(x1)
            x1 = self.conv3_2_3x3_bn(x1)
            x1 = self.conv3_2_3x3_relu(x1)

            x1 = self.conv3_2_1x1_increase(x1)
            x1 = self.conv3_2_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv3_2_relu(x)

            # conv3_3
            x1 = self.conv3_3_1x1_reduce(x)
            x1 = self.conv3_3_1x1_reduce_bn(x1)
            x1 = self.conv3_3_1x1_reduce_relu(x1)

            x1 = self.conv3_3_3x3(x1)
            x1 = self.conv3_3_3x3_bn(x1)
            x1 = self.conv3_3_3x3_relu(x1)

            x1 = self.conv3_3_1x1_increase(x1)
            x1 = self.conv3_3_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv3_3_relu(x)

            ### conv3_4
            x1 = self.conv3_4_1x1_reduce(x)
            x1 = self.conv3_4_1x1_reduce_bn(x1)
            x1 = self.conv3_4_1x1_reduce_relu(x1)

            x1 = self.conv3_4_3x3(x1)
            x1 = self.conv3_4_3x3_bn(x1)
            x1 = self.conv3_4_3x3_relu(x1)

            x1 = self.conv3_4_1x1_increase(x1)
            x1 = self.conv3_4_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv3_4_relu(x)
            # print("conv3", x.size())
            # self.conv3_out = x.clone()
            ### conv4_1 (reduce)
            x1 = self.conv4_1_1x1_reduce(x)
            x1 = self.conv4_1_1x1_reduce_bn(x1)
            x1 = self.conv4_1_1x1_reduce_relu(x1)

            x1 = self.conv4_1_3x3(x1)
            x1 = self.conv4_1_3x3_bn(x1)
            x1 = self.conv4_1_3x3_relu(x1)

            x1 = self.conv4_1_1x1_increase(x1)
            x1 = self.conv4_1_1x1_increase_bn(x1)

            # proj skip
            x2 = self.conv4_1_1x1_proj(x)
            x2 = self.conv4_1_1x1_proj_bn(x2)

            x = x1 + x2
            x = self.conv4_1_relu(x)

            ### conv4_2
            x1 = self.conv4_2_1x1_reduce(x)
            x1 = self.conv4_2_1x1_reduce_bn(x1)
            x1 = self.conv4_2_1x1_reduce_relu(x1)

            x1 = self.conv4_2_3x3(x1)
            x1 = self.conv4_2_3x3_bn(x1)
            x1 = self.conv4_2_3x3_relu(x1)

            x1 = self.conv4_2_1x1_increase(x1)
            x1 = self.conv4_2_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_2_relu(x)

            ### conv4_3
            x1 = self.conv4_3_1x1_reduce(x)
            x1 = self.conv4_3_1x1_reduce_bn(x1)
            x1 = self.conv4_3_1x1_reduce_relu(x1)

            x1 = self.conv4_3_3x3(x1)
            x1 = self.conv4_3_3x3_bn(x1)
            x1 = self.conv4_3_3x3_relu(x1)

            x1 = self.conv4_3_1x1_increase(x1)
            x1 = self.conv4_3_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_3_relu(x)

            ### conv4_4
            x1 = self.conv4_4_1x1_reduce(x)
            x1 = self.conv4_4_1x1_reduce_bn(x1)
            x1 = self.conv4_4_1x1_reduce_relu(x1)

            x1 = self.conv4_4_3x3(x1)
            x1 = self.conv4_4_3x3_bn(x1)
            x1 = self.conv4_4_3x3_relu(x1)

            x1 = self.conv4_4_1x1_increase(x1)
            x1 = self.conv4_4_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_4_relu(x)
            # self.conv4_4_out = x.clone()
            ### conv4_5
            x1 = self.conv4_5_1x1_reduce(x)
            x1 = self.conv4_5_1x1_reduce_bn(x1)
            x1 = self.conv4_5_1x1_reduce_relu(x1)

            x1 = self.conv4_5_3x3(x1)
            x1 = self.conv4_5_3x3_bn(x1)
            x1 = self.conv4_5_3x3_relu(x1)

            x1 = self.conv4_5_1x1_increase(x1)
            x1 = self.conv4_5_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_5_relu(x)

            ### conv4_6
            x1 = self.conv4_6_1x1_reduce(x)
            x1 = self.conv4_6_1x1_reduce_bn(x1)
            x1 = self.conv4_6_1x1_reduce_relu(x1)

            x1 = self.conv4_6_3x3(x1)
            x1 = self.conv4_6_3x3_bn(x1)
            x1 = self.conv4_6_3x3_relu(x1)

            x1 = self.conv4_6_1x1_increase(x1)
            x1 = self.conv4_6_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_6_relu(x)

            ### conv4_7
            x1 = self.conv4_7_1x1_reduce(x)
            x1 = self.conv4_7_1x1_reduce_bn(x1)
            x1 = self.conv4_7_1x1_reduce_relu(x1)

            x1 = self.conv4_7_3x3(x1)
            x1 = self.conv4_7_3x3_bn(x1)
            x1 = self.conv4_7_3x3_relu(x1)

            x1 = self.conv4_7_1x1_increase(x1)
            x1 = self.conv4_7_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_7_relu(x)

            ## conv4_8
            x1 = self.conv4_8_1x1_reduce(x)
            x1 = self.conv4_8_1x1_reduce_bn(x1)
            x1 = self.conv4_8_1x1_reduce_relu(x1)

            x1 = self.conv4_8_3x3(x1)
            x1 = self.conv4_8_3x3_bn(x1)
            x1 = self.conv4_8_3x3_relu(x1)

            x1 = self.conv4_8_1x1_increase(x1)
            x1 = self.conv4_8_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_8_relu(x)
            # self.conv4_8_out = x.clone()
            ### conv4_9
            x1 = self.conv4_9_1x1_reduce(x)
            x1 = self.conv4_9_1x1_reduce_bn(x1)
            x1 = self.conv4_9_1x1_reduce_relu(x1)

            x1 = self.conv4_9_3x3(x1)
            x1 = self.conv4_9_3x3_bn(x1)
            x1 = self.conv4_9_3x3_relu(x1)

            x1 = self.conv4_9_1x1_increase(x1)
            x1 = self.conv4_9_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_9_relu(x)

            ### conv4_10
            x1 = self.conv4_10_1x1_reduce(x)
            x1 = self.conv4_10_1x1_reduce_bn(x1)
            x1 = self.conv4_10_1x1_reduce_relu(x1)

            x1 = self.conv4_10_3x3(x1)
            x1 = self.conv4_10_3x3_bn(x1)
            x1 = self.conv4_10_3x3_relu(x1)

            x1 = self.conv4_10_1x1_increase(x1)
            x1 = self.conv4_10_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_10_relu(x)

            ### conv4_11
            x1 = self.conv4_11_1x1_reduce(x)
            x1 = self.conv4_11_1x1_reduce_bn(x1)
            x1 = self.conv4_11_1x1_reduce_relu(x1)

            x1 = self.conv4_11_3x3(x1)
            x1 = self.conv4_11_3x3_bn(x1)
            x1 = self.conv4_11_3x3_relu(x1)

            x1 = self.conv4_11_1x1_increase(x1)
            x1 = self.conv4_11_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_11_relu(x)

            ### conv4_12
            x1 = self.conv4_12_1x1_reduce(x)
            x1 = self.conv4_12_1x1_reduce_bn(x1)
            x1 = self.conv4_12_1x1_reduce_relu(x1)

            x1 = self.conv4_12_3x3(x1)
            x1 = self.conv4_12_3x3_bn(x1)
            x1 = self.conv4_12_3x3_relu(x1)

            x1 = self.conv4_12_1x1_increase(x1)
            x1 = self.conv4_12_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_12_relu(x)
            # self.conv4_12_out = x.clone()
            ### conv4_13
            x1 = self.conv4_13_1x1_reduce(x)
            x1 = self.conv4_13_1x1_reduce_bn(x1)
            x1 = self.conv4_13_1x1_reduce_relu(x1)

            x1 = self.conv4_13_3x3(x1)
            x1 = self.conv4_13_3x3_bn(x1)
            x1 = self.conv4_13_3x3_relu(x1)

            x1 = self.conv4_13_1x1_increase(x1)
            x1 = self.conv4_13_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_13_relu(x)

            ### conv4_14
            x1 = self.conv4_14_1x1_reduce(x)
            x1 = self.conv4_14_1x1_reduce_bn(x1)
            x1 = self.conv4_14_1x1_reduce_relu(x1)

            x1 = self.conv4_14_3x3(x1)
            x1 = self.conv4_14_3x3_bn(x1)
            x1 = self.conv4_14_3x3_relu(x1)

            x1 = self.conv4_14_1x1_increase(x1)
            x1 = self.conv4_14_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_14_relu(x)

            ### conv4_15
            x1 = self.conv4_15_1x1_reduce(x)
            x1 = self.conv4_15_1x1_reduce_bn(x1)
            x1 = self.conv4_15_1x1_reduce_relu(x1)

            x1 = self.conv4_15_3x3(x1)
            x1 = self.conv4_15_3x3_bn(x1)
            x1 = self.conv4_15_3x3_relu(x1)

            x1 = self.conv4_15_1x1_increase(x1)
            x1 = self.conv4_15_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_15_relu(x)

            ### conv4_16
            x1 = self.conv4_16_1x1_reduce(x)
            x1 = self.conv4_16_1x1_reduce_bn(x1)
            x1 = self.conv4_16_1x1_reduce_relu(x1)

            x1 = self.conv4_16_3x3(x1)
            x1 = self.conv4_16_3x3_bn(x1)
            x1 = self.conv4_16_3x3_relu(x1)

            x1 = self.conv4_16_1x1_increase(x1)
            x1 = self.conv4_16_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_16_relu(x)
            # self.conv4_16_out = x.clone()
            ### conv4_17
            x1 = self.conv4_17_1x1_reduce(x)
            x1 = self.conv4_17_1x1_reduce_bn(x1)
            x1 = self.conv4_17_1x1_reduce_relu(x1)

            x1 = self.conv4_17_3x3(x1)
            x1 = self.conv4_17_3x3_bn(x1)
            x1 = self.conv4_17_3x3_relu(x1)

            x1 = self.conv4_17_1x1_increase(x1)
            x1 = self.conv4_17_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_17_relu(x)

            ### conv4_18
            x1 = self.conv4_18_1x1_reduce(x)
            x1 = self.conv4_18_1x1_reduce_bn(x1)
            x1 = self.conv4_18_1x1_reduce_relu(x1)

            x1 = self.conv4_18_3x3(x1)
            x1 = self.conv4_18_3x3_bn(x1)
            x1 = self.conv4_18_3x3_relu(x1)

            x1 = self.conv4_18_1x1_increase(x1)
            x1 = self.conv4_18_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_18_relu(x)

            ### conv4_19
            x1 = self.conv4_19_1x1_reduce(x)
            x1 = self.conv4_19_1x1_reduce_bn(x1)
            x1 = self.conv4_19_1x1_reduce_relu(x1)

            x1 = self.conv4_19_3x3(x1)
            x1 = self.conv4_19_3x3_bn(x1)
            x1 = self.conv4_19_3x3_relu(x1)

            x1 = self.conv4_19_1x1_increase(x1)
            x1 = self.conv4_19_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_19_relu(x)

            ### conv4_20
            x1 = self.conv4_20_1x1_reduce(x)
            x1 = self.conv4_20_1x1_reduce_bn(x1)
            x1 = self.conv4_20_1x1_reduce_relu(x1)

            x1 = self.conv4_20_3x3(x1)
            x1 = self.conv4_20_3x3_bn(x1)
            x1 = self.conv4_20_3x3_relu(x1)

            x1 = self.conv4_20_1x1_increase(x1)
            x1 = self.conv4_20_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_20_relu(x)
            # self.conv4_20_out = x.clone()

            ### conv4_21
            x1 = self.conv4_21_1x1_reduce(x)
            x1 = self.conv4_21_1x1_reduce_bn(x1)
            x1 = self.conv4_21_1x1_reduce_relu(x1)

            x1 = self.conv4_21_3x3(x1)
            x1 = self.conv4_21_3x3_bn(x1)
            x1 = self.conv4_21_3x3_relu(x1)

            x1 = self.conv4_21_1x1_increase(x1)
            x1 = self.conv4_21_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_21_relu(x)

            ### conv4_22
            x1 = self.conv4_22_1x1_reduce(x)
            x1 = self.conv4_22_1x1_reduce_bn(x1)
            x1 = self.conv4_22_1x1_reduce_relu(x1)

            x1 = self.conv4_22_3x3(x1)
            x1 = self.conv4_22_3x3_bn(x1)
            x1 = self.conv4_22_3x3_relu(x1)

            x1 = self.conv4_22_1x1_increase(x1)
            x1 = self.conv4_22_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_22_relu(x)

            ### conv4_23
            x1 = self.conv4_23_1x1_reduce(x)
            x1 = self.conv4_23_1x1_reduce_bn(x1)
            x1 = self.conv4_23_1x1_reduce_relu(x1)

            x1 = self.conv4_23_3x3(x1)
            x1 = self.conv4_23_3x3_bn(x1)
            x1 = self.conv4_23_3x3_relu(x1)

            x1 = self.conv4_23_1x1_increase(x1)
            x1 = self.conv4_23_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv4_23_relu(x)
            # print("conv4", x.size())
            # self.conv4_23_out = x.clone()

            ### conv5_1 (reduce)
            x1 = self.conv5_1_1x1_reduce(x)
            x1 = self.conv5_1_1x1_reduce_bn(x1)
            x1 = self.conv5_1_1x1_reduce_relu(x1)

            x1 = self.conv5_1_3x3(x1)
            x1 = self.conv5_1_3x3_bn(x1)
            x1 = self.conv5_1_3x3_relu(x1)

            x1 = self.conv5_1_1x1_increase(x1)
            x1 = self.conv5_1_1x1_increase_bn(x1)

            # proj skip
            x2 = self.conv5_1_1x1_proj(x)
            x2 = self.conv5_1_1x1_proj_bn(x2)

            x = x1 + x2
            x = self.conv5_1_relu(x)

            ### conv5_2
            x1 = self.conv5_2_1x1_reduce(x)
            x1 = self.conv5_2_1x1_reduce_bn(x1)
            x1 = self.conv5_2_1x1_reduce_relu(x1)

            x1 = self.conv5_2_3x3(x1)
            x1 = self.conv5_2_3x3_bn(x1)
            x1 = self.conv5_2_3x3_relu(x1)

            x1 = self.conv5_2_1x1_increase(x1)
            x1 = self.conv5_2_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv5_2_relu(x)

            ### conv5_3
            x1 = self.conv5_3_1x1_reduce(x)
            x1 = self.conv5_3_1x1_reduce_bn(x1)
            x1 = self.conv5_3_1x1_reduce_relu(x1)

            x1 = self.conv5_3_3x3(x1)
            x1 = self.conv5_3_3x3_bn(x1)
            x1 = self.conv5_3_3x3_relu(x1)

            x1 = self.conv5_3_1x1_increase(x1)
            x1 = self.conv5_3_1x1_increase_bn(x1)

            x = x + x1
            x = self.conv5_3_relu(x)
            # End ResNet
            # self.resnet_out = x.clone()

            # ASPP
            # Full Image Encoder
            # print("before avg pool", x.size())
            x1 = self.reduce_pooling(x)
            # print("after", x1.size())
            x1 = self.drop_reduce(x1)
            # print(x1.size())
            x1 = x1.reshape(x1.size(0), -1)
        return x1.size(1)

    def forward(self, x):
        # print("input", x.size())
        # self.input_ = x.clone()
        # Resnet
        ### conv1
        x = self.conv1_1_3x3_s2(x)
        # self.first_conv = x.clone()
        x = self.conv1_1_3x3_s2_bn(x)
        x = self.conv1_1_3x3_s2_relu(x)

        x = self.conv1_2_3x3(x)
        x = self.conv1_2_3x3_bn(x)
        x = self.conv1_2_3x3_relu(x)

        x = self.conv1_3_3x3(x)
        x = self.conv1_3_3x3_bn(x)
        x = self.conv1_3_3x3_relu(x)

        x = self.pool1_3x3_s2(x)
        # self.conv1_out = x.clone()
        ### conv2_1 (reduce)
        x1 = self.conv2_1_1x1_reduce(x)
        x1 = self.conv2_1_1x1_reduce_bn(x1)
        x1 = self.conv2_1_1x1_reduce_relu(x1)

        x1 = self.conv2_1_3x3(x1)
        x1 = self.conv2_1_3x3_bn(x1)
        x1 = self.conv2_1_3x3_relu(x1)

        x1 = self.conv2_1_1x1_increase(x1)
        x1 = self.conv2_1_1x1_increase_bn(x1)

        # proj skip
        x2 = self.conv2_1_1x1_proj(x)
        x2 = self.conv2_1_1x1_proj_bn(x2)

        x = x1 + x2
        x = self.conv2_1_relu(x)
        # print("conv2", x.size())
        ### conv2_2
        x1 = self.conv2_2_1x1_reduce(x)
        x1 = self.conv2_2_1x1_reduce_bn(x1)
        x1 = self.conv2_2_1x1_reduce_relu(x1)

        x1 = self.conv2_2_3x3(x1)
        x1 = self.conv2_2_3x3_bn(x1)
        x1 = self.conv2_2_3x3_relu(x1)

        x1 = self.conv2_2_1x1_increase(x1)
        x1 = self.conv2_2_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv2_2_relu(x)

        ### conv2 3
        x1 = self.conv2_3_1x1_reduce(x)
        x1 = self.conv2_3_1x1_reduce_bn(x1)
        x1 = self.conv2_3_1x1_reduce_relu(x1)

        x1 = self.conv2_3_3x3(x1)
        x1 = self.conv2_3_3x3_bn(x1)
        x1 = self.conv2_3_3x3_relu(x1)

        x1 = self.conv2_3_1x1_increase(x1)
        x1 = self.conv2_3_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv2_3_relu(x)
        # self.conv2_out = x.clone()

        ### conv3_1 (reduce)
        x1 = self.conv3_1_1x1_reduce(x)
        x1 = self.conv3_1_1x1_reduce_bn(x1)
        x1 = self.conv3_1_1x1_reduce_relu(x1)

        x1 = self.conv3_1_3x3(x1)
        x1 = self.conv3_1_3x3_bn(x1)
        x1 = self.conv3_1_3x3_relu(x1)

        x1 = self.conv3_1_1x1_increase(x1)
        x1 = self.conv3_1_1x1_increase_bn(x1)

        # proj skip
        x2 = self.conv3_1_1x1_proj(x)
        x2 = self.conv3_1_1x1_proj_bn(x2)

        x = x1 + x2
        x = self.conv3_1_relu(x)
        
        ### conv3_2
        x1 = self.conv3_2_1x1_reduce(x)
        x1 = self.conv3_2_1x1_reduce_bn(x1)
        x1 = self.conv3_2_1x1_reduce_relu(x1)

        x1 = self.conv3_2_3x3(x1)
        x1 = self.conv3_2_3x3_bn(x1)
        x1 = self.conv3_2_3x3_relu(x1)

        x1 = self.conv3_2_1x1_increase(x1)
        x1 = self.conv3_2_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv3_2_relu(x)

        # conv3_3
        x1 = self.conv3_3_1x1_reduce(x)
        x1 = self.conv3_3_1x1_reduce_bn(x1)
        x1 = self.conv3_3_1x1_reduce_relu(x1)

        x1 = self.conv3_3_3x3(x1)
        x1 = self.conv3_3_3x3_bn(x1)
        x1 = self.conv3_3_3x3_relu(x1)

        x1 = self.conv3_3_1x1_increase(x1)
        x1 = self.conv3_3_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv3_3_relu(x)

        ### conv3_4
        x1 = self.conv3_4_1x1_reduce(x)
        x1 = self.conv3_4_1x1_reduce_bn(x1)
        x1 = self.conv3_4_1x1_reduce_relu(x1)

        x1 = self.conv3_4_3x3(x1)
        x1 = self.conv3_4_3x3_bn(x1)
        x1 = self.conv3_4_3x3_relu(x1)

        x1 = self.conv3_4_1x1_increase(x1)
        x1 = self.conv3_4_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv3_4_relu(x)
        # print("conv3", x.size())
        # self.conv3_out = x.clone()
        ### conv4_1 (reduce)
        x1 = self.conv4_1_1x1_reduce(x)
        x1 = self.conv4_1_1x1_reduce_bn(x1)
        x1 = self.conv4_1_1x1_reduce_relu(x1)

        x1 = self.conv4_1_3x3(x1)
        x1 = self.conv4_1_3x3_bn(x1)
        x1 = self.conv4_1_3x3_relu(x1)

        x1 = self.conv4_1_1x1_increase(x1)
        x1 = self.conv4_1_1x1_increase_bn(x1)

        # proj skip
        x2 = self.conv4_1_1x1_proj(x)
        x2 = self.conv4_1_1x1_proj_bn(x2)

        x = x1 + x2
        x = self.conv4_1_relu(x)

        ### conv4_2
        x1 = self.conv4_2_1x1_reduce(x)
        x1 = self.conv4_2_1x1_reduce_bn(x1)
        x1 = self.conv4_2_1x1_reduce_relu(x1)

        x1 = self.conv4_2_3x3(x1)
        x1 = self.conv4_2_3x3_bn(x1)
        x1 = self.conv4_2_3x3_relu(x1)

        x1 = self.conv4_2_1x1_increase(x1)
        x1 = self.conv4_2_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_2_relu(x)

        ### conv4_3
        x1 = self.conv4_3_1x1_reduce(x)
        x1 = self.conv4_3_1x1_reduce_bn(x1)
        x1 = self.conv4_3_1x1_reduce_relu(x1)

        x1 = self.conv4_3_3x3(x1)
        x1 = self.conv4_3_3x3_bn(x1)
        x1 = self.conv4_3_3x3_relu(x1)

        x1 = self.conv4_3_1x1_increase(x1)
        x1 = self.conv4_3_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_3_relu(x)

        ### conv4_4
        x1 = self.conv4_4_1x1_reduce(x)
        x1 = self.conv4_4_1x1_reduce_bn(x1)
        x1 = self.conv4_4_1x1_reduce_relu(x1)

        x1 = self.conv4_4_3x3(x1)
        x1 = self.conv4_4_3x3_bn(x1)
        x1 = self.conv4_4_3x3_relu(x1)

        x1 = self.conv4_4_1x1_increase(x1)
        x1 = self.conv4_4_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_4_relu(x)
        # self.conv4_4_out = x.clone()
        ### conv4_5
        x1 = self.conv4_5_1x1_reduce(x)
        x1 = self.conv4_5_1x1_reduce_bn(x1)
        x1 = self.conv4_5_1x1_reduce_relu(x1)

        x1 = self.conv4_5_3x3(x1)
        x1 = self.conv4_5_3x3_bn(x1)
        x1 = self.conv4_5_3x3_relu(x1)

        x1 = self.conv4_5_1x1_increase(x1)
        x1 = self.conv4_5_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_5_relu(x)

        ### conv4_6
        x1 = self.conv4_6_1x1_reduce(x)
        x1 = self.conv4_6_1x1_reduce_bn(x1)
        x1 = self.conv4_6_1x1_reduce_relu(x1)

        x1 = self.conv4_6_3x3(x1)
        x1 = self.conv4_6_3x3_bn(x1)
        x1 = self.conv4_6_3x3_relu(x1)

        x1 = self.conv4_6_1x1_increase(x1)
        x1 = self.conv4_6_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_6_relu(x)

        ### conv4_7
        x1 = self.conv4_7_1x1_reduce(x)
        x1 = self.conv4_7_1x1_reduce_bn(x1)
        x1 = self.conv4_7_1x1_reduce_relu(x1)

        x1 = self.conv4_7_3x3(x1)
        x1 = self.conv4_7_3x3_bn(x1)
        x1 = self.conv4_7_3x3_relu(x1)

        x1 = self.conv4_7_1x1_increase(x1)
        x1 = self.conv4_7_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_7_relu(x)

        ## conv4_8
        x1 = self.conv4_8_1x1_reduce(x)
        x1 = self.conv4_8_1x1_reduce_bn(x1)
        x1 = self.conv4_8_1x1_reduce_relu(x1)

        x1 = self.conv4_8_3x3(x1)
        x1 = self.conv4_8_3x3_bn(x1)
        x1 = self.conv4_8_3x3_relu(x1)

        x1 = self.conv4_8_1x1_increase(x1)
        x1 = self.conv4_8_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_8_relu(x)
        # self.conv4_8_out = x.clone()
        ### conv4_9
        x1 = self.conv4_9_1x1_reduce(x)
        x1 = self.conv4_9_1x1_reduce_bn(x1)
        x1 = self.conv4_9_1x1_reduce_relu(x1)

        x1 = self.conv4_9_3x3(x1)
        x1 = self.conv4_9_3x3_bn(x1)
        x1 = self.conv4_9_3x3_relu(x1)

        x1 = self.conv4_9_1x1_increase(x1)
        x1 = self.conv4_9_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_9_relu(x)

        ### conv4_10
        x1 = self.conv4_10_1x1_reduce(x)
        x1 = self.conv4_10_1x1_reduce_bn(x1)
        x1 = self.conv4_10_1x1_reduce_relu(x1)

        x1 = self.conv4_10_3x3(x1)
        x1 = self.conv4_10_3x3_bn(x1)
        x1 = self.conv4_10_3x3_relu(x1)

        x1 = self.conv4_10_1x1_increase(x1)
        x1 = self.conv4_10_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_10_relu(x)

        ### conv4_11
        x1 = self.conv4_11_1x1_reduce(x)
        x1 = self.conv4_11_1x1_reduce_bn(x1)
        x1 = self.conv4_11_1x1_reduce_relu(x1)

        x1 = self.conv4_11_3x3(x1)
        x1 = self.conv4_11_3x3_bn(x1)
        x1 = self.conv4_11_3x3_relu(x1)

        x1 = self.conv4_11_1x1_increase(x1)
        x1 = self.conv4_11_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_11_relu(x)

        ### conv4_12
        x1 = self.conv4_12_1x1_reduce(x)
        x1 = self.conv4_12_1x1_reduce_bn(x1)
        x1 = self.conv4_12_1x1_reduce_relu(x1)

        x1 = self.conv4_12_3x3(x1)
        x1 = self.conv4_12_3x3_bn(x1)
        x1 = self.conv4_12_3x3_relu(x1)

        x1 = self.conv4_12_1x1_increase(x1)
        x1 = self.conv4_12_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_12_relu(x)
        # self.conv4_12_out = x.clone()
        ### conv4_13
        x1 = self.conv4_13_1x1_reduce(x)
        x1 = self.conv4_13_1x1_reduce_bn(x1)
        x1 = self.conv4_13_1x1_reduce_relu(x1)

        x1 = self.conv4_13_3x3(x1)
        x1 = self.conv4_13_3x3_bn(x1)
        x1 = self.conv4_13_3x3_relu(x1)

        x1 = self.conv4_13_1x1_increase(x1)
        x1 = self.conv4_13_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_13_relu(x)

        ### conv4_14
        x1 = self.conv4_14_1x1_reduce(x)
        x1 = self.conv4_14_1x1_reduce_bn(x1)
        x1 = self.conv4_14_1x1_reduce_relu(x1)

        x1 = self.conv4_14_3x3(x1)
        x1 = self.conv4_14_3x3_bn(x1)
        x1 = self.conv4_14_3x3_relu(x1)

        x1 = self.conv4_14_1x1_increase(x1)
        x1 = self.conv4_14_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_14_relu(x)

        ### conv4_15
        x1 = self.conv4_15_1x1_reduce(x)
        x1 = self.conv4_15_1x1_reduce_bn(x1)
        x1 = self.conv4_15_1x1_reduce_relu(x1)

        x1 = self.conv4_15_3x3(x1)
        x1 = self.conv4_15_3x3_bn(x1)
        x1 = self.conv4_15_3x3_relu(x1)

        x1 = self.conv4_15_1x1_increase(x1)
        x1 = self.conv4_15_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_15_relu(x)

        ### conv4_16
        x1 = self.conv4_16_1x1_reduce(x)
        x1 = self.conv4_16_1x1_reduce_bn(x1)
        x1 = self.conv4_16_1x1_reduce_relu(x1)

        x1 = self.conv4_16_3x3(x1)
        x1 = self.conv4_16_3x3_bn(x1)
        x1 = self.conv4_16_3x3_relu(x1)

        x1 = self.conv4_16_1x1_increase(x1)
        x1 = self.conv4_16_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_16_relu(x)
        # self.conv4_16_out = x.clone()
        ### conv4_17
        x1 = self.conv4_17_1x1_reduce(x)
        x1 = self.conv4_17_1x1_reduce_bn(x1)
        x1 = self.conv4_17_1x1_reduce_relu(x1)

        x1 = self.conv4_17_3x3(x1)
        x1 = self.conv4_17_3x3_bn(x1)
        x1 = self.conv4_17_3x3_relu(x1)

        x1 = self.conv4_17_1x1_increase(x1)
        x1 = self.conv4_17_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_17_relu(x)

        ### conv4_18
        x1 = self.conv4_18_1x1_reduce(x)
        x1 = self.conv4_18_1x1_reduce_bn(x1)
        x1 = self.conv4_18_1x1_reduce_relu(x1)

        x1 = self.conv4_18_3x3(x1)
        x1 = self.conv4_18_3x3_bn(x1)
        x1 = self.conv4_18_3x3_relu(x1)

        x1 = self.conv4_18_1x1_increase(x1)
        x1 = self.conv4_18_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_18_relu(x)

        ### conv4_19
        x1 = self.conv4_19_1x1_reduce(x)
        x1 = self.conv4_19_1x1_reduce_bn(x1)
        x1 = self.conv4_19_1x1_reduce_relu(x1)

        x1 = self.conv4_19_3x3(x1)
        x1 = self.conv4_19_3x3_bn(x1)
        x1 = self.conv4_19_3x3_relu(x1)

        x1 = self.conv4_19_1x1_increase(x1)
        x1 = self.conv4_19_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_19_relu(x)

        ### conv4_20
        x1 = self.conv4_20_1x1_reduce(x)
        x1 = self.conv4_20_1x1_reduce_bn(x1)
        x1 = self.conv4_20_1x1_reduce_relu(x1)

        x1 = self.conv4_20_3x3(x1)
        x1 = self.conv4_20_3x3_bn(x1)
        x1 = self.conv4_20_3x3_relu(x1)

        x1 = self.conv4_20_1x1_increase(x1)
        x1 = self.conv4_20_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_20_relu(x)
        # self.conv4_20_out = x.clone()

        ### conv4_21
        x1 = self.conv4_21_1x1_reduce(x)
        x1 = self.conv4_21_1x1_reduce_bn(x1)
        x1 = self.conv4_21_1x1_reduce_relu(x1)

        x1 = self.conv4_21_3x3(x1)
        x1 = self.conv4_21_3x3_bn(x1)
        x1 = self.conv4_21_3x3_relu(x1)

        x1 = self.conv4_21_1x1_increase(x1)
        x1 = self.conv4_21_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_21_relu(x)

        ### conv4_22
        x1 = self.conv4_22_1x1_reduce(x)
        x1 = self.conv4_22_1x1_reduce_bn(x1)
        x1 = self.conv4_22_1x1_reduce_relu(x1)

        x1 = self.conv4_22_3x3(x1)
        x1 = self.conv4_22_3x3_bn(x1)
        x1 = self.conv4_22_3x3_relu(x1)

        x1 = self.conv4_22_1x1_increase(x1)
        x1 = self.conv4_22_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_22_relu(x)

        ### conv4_23
        x1 = self.conv4_23_1x1_reduce(x)
        x1 = self.conv4_23_1x1_reduce_bn(x1)
        x1 = self.conv4_23_1x1_reduce_relu(x1)

        x1 = self.conv4_23_3x3(x1)
        x1 = self.conv4_23_3x3_bn(x1)
        x1 = self.conv4_23_3x3_relu(x1)

        x1 = self.conv4_23_1x1_increase(x1)
        x1 = self.conv4_23_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv4_23_relu(x)
        # print("conv4", x.size())
        # self.conv4_23_out = x.clone()

        ### conv5_1 (reduce)
        x1 = self.conv5_1_1x1_reduce(x)
        x1 = self.conv5_1_1x1_reduce_bn(x1)
        x1 = self.conv5_1_1x1_reduce_relu(x1)

        x1 = self.conv5_1_3x3(x1)
        x1 = self.conv5_1_3x3_bn(x1)
        x1 = self.conv5_1_3x3_relu(x1)

        x1 = self.conv5_1_1x1_increase(x1)
        x1 = self.conv5_1_1x1_increase_bn(x1)

        # proj skip
        x2 = self.conv5_1_1x1_proj(x)
        x2 = self.conv5_1_1x1_proj_bn(x2)

        x = x1 + x2
        x = self.conv5_1_relu(x)

        ### conv5_2
        x1 = self.conv5_2_1x1_reduce(x)
        x1 = self.conv5_2_1x1_reduce_bn(x1)
        x1 = self.conv5_2_1x1_reduce_relu(x1)

        x1 = self.conv5_2_3x3(x1)
        x1 = self.conv5_2_3x3_bn(x1)
        x1 = self.conv5_2_3x3_relu(x1)

        x1 = self.conv5_2_1x1_increase(x1)
        x1 = self.conv5_2_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv5_2_relu(x)

        ### conv5_3
        x1 = self.conv5_3_1x1_reduce(x)
        x1 = self.conv5_3_1x1_reduce_bn(x1)
        x1 = self.conv5_3_1x1_reduce_relu(x1)

        x1 = self.conv5_3_3x3(x1)
        x1 = self.conv5_3_3x3_bn(x1)
        x1 = self.conv5_3_3x3_relu(x1)

        x1 = self.conv5_3_1x1_increase(x1)
        x1 = self.conv5_3_1x1_increase_bn(x1)

        x = x + x1
        x = self.conv5_3_relu(x)
        # End ResNet
        # self.resnet_out = x.clone()

        # ASPP
        # Full Image Encoder
        # print("before avg pool", x.size())
        x1 = self.reduce_pooling(x)
        # print("after", x1.size())
        x1 = self.drop_reduce(x1)
        # print(x1.size())
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.ip1_depth(x1)
        x1 = self.relu_ip1(x1)
        x1 = x1.unsqueeze(-1).unsqueeze(-1)

        x1 = self.conv6_1_soft(x1)
        x1 = self.relu6_1(x1)
        # End Full Image Encoder
        # self.encoder_out = x1.clone()
        # ASPP 1x1 conv
        x2 = self.aspp_1_soft(x)
        x2 = self.relu_aspp_1(x2)
        x2 = self.conv6_2_soft(x2)
        x2 = self.relu6_2(x2)
        # End ASPP 1x1 conv
        # self.aspp2_out = x2.clone()
        # ASPP dilation 4
        x3 = self.aspp_2_soft(x)
        x3 = self.relu_aspp_2(x3)
        x3 = self.conv6_3_soft(x3)
        x3 = self.relu6_3(x3)
        # End ASPP dilation 4
        # self.aspp3_out = x3.clone()
        # ASPP dilation 8
        x4 = self.aspp_3_soft(x)
        x4 = self.relu_aspp_3(x4)
        x4 = self.conv6_4_soft(x4)
        x4 = self.relu6_4(x4)
        # End ASPP dilation 8
        # self.aspp4_out = x4.clone()
        # ASPP dilation 12
        x5 = self.aspp_4_soft(x)
        x5 = self.relu_aspp_4(x5)
        x5 = self.conv6_5_soft(x5)
        x5 = self.relu6_5(x5)
        # End ASPP dilation 12
        # self.aspp5_out = x5.clone()

        # Concatenate
        x = torch.cat([x1.expand(-1, -1, x2.size(2), x2.size(3)), x2, x3, x4, x5], dim=1)

        x = self.drop_conv6(x)
        x = self.conv7_soft(x)
        x = self.relu7(x)
        x = self.drop_conv7(x)

        x = self.conv8(x)
        # print("before zoom", x.size())
        # Upsample by a factor of 8 using bilinear interpolation
        zoom_factor = 8
        height_out = x.size(2) + (x.size(2)-1) * (zoom_factor-1)
        width_out = x.size(3) + (x.size(3)-1) * (zoom_factor-1)
        x = F.interpolate(x, size=(height_out, width_out), mode="bilinear", align_corners=False)

        return x

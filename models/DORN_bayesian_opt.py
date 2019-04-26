import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np
from models.data.utils.sid_utils import SIDTorch
from torch.optim import SGD
import matplotlib
# matplotlib.use("TKAgg")
# import matplotlib.pyplot as plt
import os

# from models.core.model_core import Model

from models.DORN_nohints import DORN_nyu_nohints

class DORN_bayesian_opt:
    """
    Performs SGD to optimize the depth map further after being given an initial depth map
    estimate from DORN.
    """
    def __init__(self, sgd_iters=1, use_albedo=True, use_squared_falloff=True, normalize_to=1000.,
                 lr=1e-3, hints_len=68, spad_weight=1.,
                 in_channels=3, in_height=257, in_width=353,
                 sid_bins=68, offset=0.,
                 min_depth=0., max_depth=10.,
                 alpha=0.6569154266167957, beta=9.972175646365525,
                 frozen=True, pretrained=True,
                 state_dict_file=os.path.join("models", "torch_params_nyuv2_BGR.pth.tar")):
        """
        :param hints_len: Uniformly spaced noisy depth hints (i.e. raw SPAD data)
        :param num_hints_layers: The number of layers for performing upsampling
        """
        self.sgd_iters = sgd_iters
        self.use_albedo = use_albedo
        self.use_squared_falloff = use_squared_falloff
        self.normalize_to = normalize_to
        self.lr = lr
        self.loss = MSELoss()
        # self.loss.to(device)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.spad_weight = spad_weight
        self.hints_len = hints_len
        self.sid_bins = sid_bins
        self.feature_extractor = \
            DORN_nyu_nohints(in_channels, in_height, in_width,
                             sid_bins, offset,
                             min_depth, max_depth,
                             alpha, beta,
                             frozen, pretrained,
                             state_dict_file)
        self.feature_extractor.eval()    # Only use DORN in eval mode.

        self.sid_obj = SIDTorch(sid_bins, alpha, beta, offset)
        self.one_over_depth_squared = 1./(self.sid_obj.sid_bin_values[:-2] ** 2)
        self.one_over_depth_squared.requires_grad = False

    def initialize(self, input_, device):
        """Feed rgb through DORN
        :return per-pixel one-hot indicating the depth bin for that pixel.
        """
        self.feature_extractor.to(device)
        rgb = input_["rgb"].to(device)
        with torch.no_grad():
            x = self.feature_extractor(rgb)
            log_probs, _ = self.to_logprobs(x)
            depth_index = torch.sum((log_probs >= np.log(0.5)), dim=1, keepdim=True).long()
            # print(depth_index[:,:,3,3])
            depth = torch.zeros_like(log_probs, device=device)

            # Add base level of uncertainty
            # base = 0.8
            # depth.add_(base)

            depth.scatter_(dim=1, index=depth_index, value=1)

            # Base level
            print(depth[:,:,1,1])
        return depth, log_probs, depth_index

    # def optimize_depth_map(self, depth_hist, input_, device):
    def optimize_depth_map(self, depth_index, input_, device):
        """Perform SGD on the initial input to get the refined input"""
        gt_hist = input_["spad"].to(device)
        albedo = input_["albedo"].to(device)
        _, _, W, H = depth_index.size()
        gt_hist *= W*H/torch.sum(gt_hist)

        plt.figure()
        plt.plot(gt_hist.clone().squeeze().detach().cpu().numpy())
        plt.title("spad")
        plt.draw()
        plt.pause(0.001)

        albedo = albedo[:,1,:,:].requires_grad_(False)
        # with torch.no_grad():
        #     depth_hist = torch.log(depth_hist)
        output = depth_index.clone().detach().float().requires_grad_(True).to(device)
        depth_bin_indices = torch.Tensor(range(self.sid_bins)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        # sid_bin_values = self.sid_obj.sid_bin_values.to(device)[:68]
        # print(output.dtype)
        optimizer = SGD([output], lr=self.lr)
        # self.zero_pad = torch.zeros((N, 1, W, H), requires_grad=False).to(device)
        self.one_over_depth_squared = self.one_over_depth_squared.to(device)
        # self.zero_pad.requires_grad = False
        for it in range(self.sgd_iters):
            # Renormalize each output pixel's histogram
            # with torch.no_grad():
            # output = output / torch.sum(output, dim=1, keepdim=True)
            # output_probs = torch.exp(output)
            # output_probs = self.depth_to_hist(output, depth_bin_indices)
            output_hist = self.spad_forward(output, albedo)

            if not it % 100:
                # plt.figure()
                # plt.plot(output_probs[:, :, 130, 130].squeeze().clone().detach().cpu().numpy())
                # plt.title("(130, 130) output_probs at iteration {}".format(it))
                # plt.draw()
                # plt.pause(0.001)

                plt.figure()
                output_spad = output_hist/torch.sum(output_hist)
                plt.plot(output_spad.squeeze().clone().detach().cpu().numpy())
                plt.title("SPAD at iteration {}".format(it))
                plt.draw()
                plt.pause(0.001)

            # plt.figure()
            # plt.plot(output[:,:,130,130].squeeze().clone().detach().cpu().numpy())
            # plt.title("(130, 130) histogram at step {}".format(it))
            # plt.draw()
            # plt.pause(0.001)

            loss_val = self.loss(gt_hist, output_hist)
            print("loss_val", loss_val)
            optimizer.zero_grad()
            loss_val.backward()
            print("output_grad", torch.norm(output.grad))
            optimizer.step()
        return output

    def spad_forward(self, depth_hist, albedo):
        """Perform the forward simulation of a model"""
        N, C, W, H = depth_hist.size()
        # probs = torch.exp(log_probs)
        # Get histogram from 1-cdf
        # probs = probs[:, :-1, :, :] - probs[:, 1:, :, :]
        # print(torch.sum(probs[0,:,0,0]))
        # probs = torch.cat([self.zero_pad, probs], dim=1)
        # Use probs to get the weighted sum of albedo/depth^2
        one_over_depth_squared_unsqueezed = self.one_over_depth_squared.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # print(probs.shape)
        # print(self.one_over_depth_squared.expand(N, C, W, H).shape)
        # print(albedo_expanded.shape)

        weights = depth_hist
        if self.use_albedo:
            weights *= albedo
        if self.use_squared_falloff:
            weights *= one_over_depth_squared_unsqueezed
        simulated_spad = torch.sum(weights, dim=(2, 3), keepdim=True)
        print("simulated scale factor: {}".format(W*H/torch.sum(simulated_spad)))
        simulated_spad *= W*H/torch.sum(simulated_spad)
        return simulated_spad

    @staticmethod
    def depth_to_hist(depth_index, depth_bin_indices):
        """Takes a depth map with indices corresponding to depth bins and outputs an
        approximate histogram from, differentiably.
        """
        # N, C, W, H = depth_index.size()
        # print(depth_index.dtype)

        # print(depth_index[:,:,130,130])
        diff = depth_index - depth_bin_indices
        # print(diff[:,:,130,130])
        depth_hist = F.softmax(1./(diff**2 + 1e-4), dim=1)
        return depth_hist


    def evaluate(self, data, device):
        depth_hist = model.initialize(data, device)
        output_depth_probs = model.optimize_depth_map(depth_hist, data, device)
        output_dir = "."
        outfile = os.path.join(output_dir, "{}_pred_{}.png".format(input_["entry"].replace("/", "_"), sgd_iters))
        output_depth = model.ord_decode((output_depth_probs, None), model.sid_obj)

DORN_bayesian_opt.to_logprobs = staticmethod(DORN_nyu_nohints.to_logprobs)
DORN_bayesian_opt.ord_decode = staticmethod(DORN_nyu_nohints.ord_decode)
DORN_bayesian_opt.get_metrics = staticmethod(DORN_nyu_nohints.get_metrics)


if __name__ == "__main__":
    import os
    import numpy as np
    from torch.utils.data import DataLoader
    from models.data.utils.sid_utils import SID
    from models.data.nyuv2_official_hints_sid_dataset import load_data, cfg
    from models.data.utils.spad_utils import SimulateSpad
    from models.data.utils.spad_utils import cfg as spad_cfg
    import torchvision.utils as vutils
    import matplotlib
    matplotlib.use("TKAgg")
    import matplotlib.pyplot as plt

    data_config = cfg()
    # print(data_config)
    # data_config["normalization"] = "dorn"
    spad_config = spad_cfg()
    spad_config["dc_count"] = 0.
    spad_config["use_albedo"] = False
    spad_config["use_squared_falloff"] = False
    # print(config)
    # print(spad_config)
    del data_config["data_name"]
    device = torch.device("cuda")
    _, val, test = load_data(**data_config, spad_config=spad_config)
    # input_ = val[1]
    input_ = test.get_item_by_id("living_room_0059/1591")
    for key in ["rgb", "albedo", "rawdepth", "spad", "mask"]:
        input_[key] = input_[key].unsqueeze(0)
    print(input_["entry"])

    # sgd_iters = 400
    sgd_iters = 300
    model = DORN_bayesian_opt(sgd_iters=sgd_iters, use_albedo=spad_config["use_albedo"],
                              use_squared_falloff=spad_config["use_squared_falloff"], lr=1e-3)
    model.feature_extractor.to(device)



    plt.figure()
    depth_truth_img = vutils.make_grid(input_["rawdepth"], normalize=True, range=(model.min_depth, model.max_depth))
    plt.imshow(depth_truth_img.numpy().transpose(1, 2, 0))
    plt.title("depth_truth")
    plt.draw()
    plt.pause(0.001)

    # plt.figure()
    # albedo_img = vutils.make_grid(input_["albedo"][:,1,:,:])
    # plt.title("albedo")
    # plt.imshow(albedo_img.numpy().transpose(1, 2, 0)/255.)
    # plt.draw()
    # plt.pause(0.001)

    # for sgd_iters in range(5):
    print("Running with {} iters of SGD...".format(sgd_iters))

    depth, log_probs_dorn, depth_index = model.initialize(input_, device)
    # log_probs_optimized = model.optimize_depth_map(depth, input_, device)
    # depth_optimized = model.optimize_depth_map(depth, input_, device)
    depth_optimized_index = model.optimize_depth_map(depth_index, input_, device)

    # Decode the output using usual ordinal decoding method.
    # log_probs_optimized = torch.log(1. - torch.cumsum(F.softmax(log_probs_optimized, dim=1), dim=1))
    #
    # plt.figure()
    # plt.plot(log_probs_optimized[:,:,130,130].squeeze().clone().detach().cpu().numpy())
    # plt.title("(130, 130) optimized log-histogram")
    # plt.draw()
    # plt.pause(0.001)

    # plt.figure()
    # plt.plot(log_probs_dorn[:, :, 130, 130].squeeze().clone().detach().cpu().numpy())
    # plt.title("(130, 130) DORN log-histogram")
    # plt.draw()
    # plt.pause(0.001)

    # log_probs = log_probs_dorn + 0.3*log_probs_optimized
    # depth_pred = model.ord_decode((log_probs,None), model.sid_obj)

    # Decode by rounding the output index and taking from the sid_obj
    depth_optimized_index_rounded = depth_optimized_index.round().detach().cpu().long()
    depth_pred = model.sid_obj.get_value_from_sid_index(depth_optimized_index_rounded)
    depth_img = vutils.make_grid(depth_pred, normalize=True, range=(model.min_depth, model.max_depth))

    plt.figure()
    plt.imshow(depth_img.numpy().transpose(1, 2, 0))
    plt.title("Predicted depth")
    plt.draw()
    plt.pause(0.001)

    # Get output histogram as well
    # print(test.transform[6])
    depth_img_np = depth_img.numpy()
    simulate_spad_transform = SimulateSpad("rawdepth", "albedo", "mask", "spad", data_config["min_depth"], data_config["max_depth"],
                                           spad_config["spad_bins"],
                                           spad_config["photon_count"],
                                           spad_config["dc_count"],
                                           spad_config["fwhm_ps"],
                                           spad_config["use_albedo"],
                                           spad_config["use_squared_falloff"],
                                           sid_obj=SID(data_config["sid_bins"], data_config["alpha"],
                                                       data_config["beta"], data_config["offset"]))
    print(depth_pred.shape)
    print(input_["albedo"].shape)
    print(input_["mask"].shape)
    sample = {"rawdepth": depth_pred.numpy().squeeze(0).transpose(1, 2, 0),
              "albedo": input_["albedo"].numpy().squeeze(0).transpose(1, 2, 0),
              "mask": np.ones_like(depth_pred.numpy()).squeeze(0).transpose(1, 2, 0)}
    depth_img_spad = simulate_spad_transform(sample)
    print(depth_img_spad["spad"])
    plt.figure()
    plt.plot(depth_img_spad["spad"])
    plt.title("Predicted depth histogram")
    plt.draw()
    plt.pause(0.001)

    # print(depth_pred.shape)
    # print(input_["rawdepth"].shape)
    # print(input_["mask"].shape)
    print(model.get_metrics(depth_pred, input_["rawdepth"], input_["mask"]))
    # print(output[:,:,30,30])
    # output_dir = "."
    # outfile = os.path.join(output_dir, "{}_pred_{}.png".format(input_["entry"].replace("/", "_"), sgd_iters))
        # output_depth = model.ord_decode((output_depth_probs, None), model.sid_obj)
        # # print(output_depth)
        #
        # vutils.save_image(output_depth, outfile, nrow=1, normalize=True, range=(model.min_depth, model.max_depth))
    input("Press the enter key to exit.")

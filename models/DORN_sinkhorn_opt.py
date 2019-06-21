import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np
from models.data.utils.sid_utils import SIDTorch
from models.sinkhorn_dist import optimize_depth_map_masked
from models.data.utils.spad_utils import remove_dc_from_spad, bgr2gray
from utils.inspect_results import add_hist_plot, log_single_gray_img
from torch.optim import SGD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pdb import set_trace

# from models.core.model_core import Model

from models.DORN_nohints import DORN_nyu_nohints
from models.sinkhorn_opt import SinkhornOpt, SinkhornOptFull

class DORN_sinkhorn_opt(SinkhornOptFull):
    """
    Performs SGD to optimize the depth map further after being given an initial depth map
    estimate from DORN.
    """
    def __init__(self, sgd_iters=250, sinkhorn_iters=40, sigma=2., lam=1e-2, kde_eps=1e-5,
                 sinkhorn_eps=1e-2, dc_eps=1e-5,
                 remove_dc=True, use_intensity=True, use_squared_falloff=True,
                 lr=1e3, hints_len=68,
                 in_channels=3, in_height=257, in_width=353,
                 sid_bins=68, offset=0.,
                 min_depth=0., max_depth=10.,
                 alpha=0.6569154266167957, beta=9.972175646365525,
                 frozen=True, pretrained=True,
                 state_dict_file=os.path.join("models", "torch_params_nyuv2_BGR.pth.tar")):
        pass

if __name__ == "__main__":
    import os
    from time import perf_counter
    import numpy as np
    from torch.utils.data import DataLoader
    from utils.train_utils import init_randomness
    from models.data.nyuv2_official_hints_sid_dataset import load_data, cfg
    from models.data.utils.spad_utils import cfg as spad_cfg
    from collections import defaultdict
    data_config = cfg()
    spad_config = spad_cfg()
    # spad_config["dc_count"] = 0.
    # spad_config["use_albedo"] = False
    # spad_config["use_squared_falloff"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(config)
    # print(spad_config)
    del data_config["data_name"]
    model = DORN_sinkhorn_opt(sgd_iters=400, sinkhorn_iters=40, sigma=.5, lam=1e-2,
                              kde_eps=1e-4, sinkhorn_eps=1e-4,
                              remove_dc=spad_config["dc_count"] > 0., use_intensity=spad_config["use_intensity"],
                              use_squared_falloff=spad_config["use_squared_falloff"],
                              lr=1e3)
    model.to(device)
    _, _, test = load_data(**data_config, spad_config=spad_config)

    dataloader = DataLoader(test, shuffle=True)
    start = perf_counter()
    init_randomness(95290421)
    input_ = test.get_item_by_id("kitchen_0002/1121")
    for key in ["rgb", "rgb_orig", "rawdepth", "spad", "mask", "rawdepth_orig", "mask_orig"]:
        input_[key] = input_[key].unsqueeze(0).to(device)
    data_load_time = perf_counter() - start
    print("dataloader: {}".format(data_load_time))
    # print(input_["entry"])
    # print(model.hints_extractor[0].weight)

    # Checks
    print(input_["entry"])
    print("remove_dc: ", model.remove_dc)
    print("use_intensity: ", model.use_intensity)
    print("use_squared_falloff: ", model.use_squared_falloff)
    pred, pred_metrics = model.evaluate(input_, device)
    print(pred_metrics)

    num_pixels = 0.
    avg_metrics = defaultdict(float)
    metrics = defaultdict(dict)
    # for i, data in enumerate(dataloader):
    #     # TESTING
    #     if i == 5:
    #         break
    #     entry = data["entry"][0]
    #     print("Evaluating {}".format(data["entry"][0]))
    #     pred, pred_metrics = model.evaluate(data, device)
    #     print(pred_metrics)
    #     metrics[entry] = pred_metrics
    #     num_valid_pixels = torch.sum(data["mask_orig"]).item()
    #     num_pixels += num_valid_pixels
    #     for metric_name in pred_metrics:
    #         avg_metrics[metric_name] += num_valid_pixels * pred_metrics[metric_name]
    #     # print(pred_metrics)
    #     # Option to save outputs:
    #     # if save_outputs:
    #     #     if output_dir is None:
    #     #         raise ValueError("evaluate_model_on_dataset: output_dir is None")
    #     #     save_dict = {
    #     #         "entry": entry,
    #     #         "pred": pred
    #     #     }
    #     #     path = os.path.join(output_dir, "{}_out.pt".format(entry))
    #     #     safe_makedir(os.path.dirname(path))
    #     #     torch.save(save_dict, path)
    #
    # for metric_name in avg_metrics:
    #     avg_metrics[metric_name] /= num_pixels
    # print(avg_metrics)

    #     with open(os.path.join(output_dir, "avg_metrics.json"), "w") as f:
    #         json.dump(avg_metrics, f)
    #     with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    #         json.dump(metrics, f)
    # # print(before_metrics)
    # print(metrics)
    # print(model.sid_obj)
    # input("press the enter key to finish.")

#!/usr/bin/env python3
import os
import torch
import json
from torch.utils.data import DataLoader
from utils.train_utils import init_randomness, worker_init_randomness
from utils.eval_utils import evaluate_model_on_dataset, evaluate_model_on_data_entry
from models.core.checkpoint import load_checkpoint, safe_makedir
from models import make_model
from sacred import Experiment
from sacred.observers import FileStorageObserver
from time import perf_counter

# Dataset
from models.data.nyuv2_test_split_dataset_hints_sid import nyuv2_test_split_ingredient, load_data
from models.data.utils.spad_utils import spad_ingredient

ex = Experiment('eval_dorn_sinkhorn_opt_test_split', ingredients=[nyuv2_test_split_ingredient, spad_ingredient])

@ex.config
def cfg(data_config, spad_config):
    model_config = {                            # Load pretrained model for testing
        "model_name": "DORN_sinkhorn_opt",
        "model_params": {
            "sgd_iters": 100,
            "sinkhorn_iters": 40,
            "sigma": 0.5,
            "lam": 1e1,
            "kde_eps": 1e-4,
            "sinkhorn_eps": 1e-7,
            "dc_eps": 1e-5,
            "remove_dc": spad_config["dc_count"] > 0.,
            "use_intensity": spad_config["use_intensity"],
            "use_squared_falloff": spad_config["use_squared_falloff"],
            "lr": 1e5,
            "hints_len": 68,
            "in_channels": 3,
            "in_height": 257,
            "in_width": 353,
            "sid_bins": data_config["sid_bins"],
            "offset": data_config["offset"],
            "min_depth": data_config["min_depth"],
            "max_depth": data_config["max_depth"],
            "alpha": data_config["alpha"],
            "beta": data_config["beta"],
            "frozen": True,
            "pretrained": True,
            "state_dict_file": os.path.join("models", "torch_params_nyuv2_BGR.pth.tar"),
        },
        "model_state_dict_fn": None             # Keep as None
    }
    ckpt_file = None                            # Keep as None
    save_outputs = True
    seed = 95290421
    small_run = 0
    entry = None
    # hyperparams = ["sgd_iters", "sinkhorn_iters", "sigma", "lam", "kde_eps", "sinkhorn_eps"]
    pdict = model_config["model_params"]
    comment = "_".join(["sgd_iters_{}".format(pdict["sgd_iters"]),
                        "sinkhorn_iters_{}".format(pdict["sinkhorn_iters"]),
                        "sigma_{}".format(pdict["sigma"]),
                        "lam_{}".format(pdict["lam"]),
                        "kde_eps_{}".format(pdict["kde_eps"]),
                        "sinkhorn_eps_{}".format(pdict["sinkhorn_eps"]),
                        ])
    del pdict
    # print(data_config.keys())
    fullcomment = comment + "_" + spad_config["spad_comment"]
    output_dir = os.path.join("results",
                              data_config["data_name"],    # e.g. nyu_depth_v2
                              "test_{}".format(small_run),
                              model_config["model_name"])  # e.g. DORN_nyu_nohints
    if fullcomment is not "":
        output_dir = os.path.join(output_dir, fullcomment)

    safe_makedir(output_dir)
    ex.observers.append(FileStorageObserver.create(os.path.join(output_dir, "runs")))


    cuda_device = "0"                       # The gpu index to run on. Should be a string
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    # print("after: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))
    if ckpt_file is not None:
        model_update, _, _ = load_checkpoint(ckpt_file)
        model_config.update(model_update)
        del model_update, _  # So sacred doesn't collect them.


@ex.automain
def main(model_config,
         save_outputs,
         output_dir,
         data_config,
         seed,
         small_run,
         entry,
         device):
    # Load the model
    model = make_model(**model_config)
    # model.sid_obj.to(device)
    # print(model)
    model.to(device)

    from tensorboardX import SummaryWriter
    from datetime import datetime
    # model.writer = SummaryWriter(log_dir=os.path.join("runs",
    #                                                   datetime.now().strftime('%b%d'),
    #                                                   datetime.now().strftime('%H-%M-%S_') + \
    #                                                   "dorn_sinkhorn_opt"))

    # Load the data
    dataset = load_data(dorn_mode=True)
    eval_fn = lambda input_, device: model.evaluate(input_["rgb_cropped"].to(device),
                                                    input_["rgb_cropped_orig"].to(device),
                                                    input_["spad"].to(device),
                                                    input_["mask_orig"].to(device),
                                                    input_["depth_cropped_orig"].to(device),
                                                    device)
    init_randomness(seed)
    if entry is None:
        print("Evaluating the model on {}.".format(data_config["data_name"]))
        evaluate_model_on_dataset(eval_fn, dataset, small_run, device, save_outputs, output_dir)
    else:
        print("Evaluating {}".format(entry))
        evaluate_model_on_data_entry(eval_fn, dataset, entry, device, save_outputs, output_dir)
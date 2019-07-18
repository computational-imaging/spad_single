#!/usr/bin/env python3
import os
import torch
from utils.train_utils import init_randomness
from utils.eval_utils import evaluate_model_on_dataset, evaluate_model_on_data_entry
from models.core.checkpoint import load_checkpoint, safe_makedir
from models import make_model
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Dataset
from models.data.nyuv2_labeled_dataset import nyuv2_labeled_ingredient, load_data

ex = Experiment('eval_dorn_nyuv2_labeled', ingredients=[nyuv2_labeled_ingredient])


# Tensorboardx
# writer = SummaryWriter()

@ex.config
def cfg(data_config):
    model_config = {                            # Load pretrained model for testing
        "model_name": "DORN_histogram_matching_wass",
        "model_params": {
            "in_channels": 3,
            "in_height": 257,
            "in_width": 353,
            "sgd_iters": 200,
            "sinkhorn_iters": 50,
            "sigma": 0.75,
            "lam": 1e1,
            "kde_eps": 1e-4,
            "sinkhorn_eps": 1e-7,
            "dc_eps": 1e-5,
            "lr": 1e-2,
            "sid_bins": 68,
            "offset": 0.,
            "min_depth": 0.,
            "max_depth": 10.,
            "alpha": 0.6569154266167957,
            "beta": 9.972175646365525,
            "frozen": True,
            "pretrained": True,
            "state_dict_file": os.path.join("models", "torch_params_nyuv2_BGR.pth.tar"),
        },
        "model_state_dict_fn": None
    }

    ckpt_file = None                            # Keep as None
    dataset_type = "test"
    save_outputs = True
    seed = 95290421
    small_run = 0
    entry = None

    # hyperparams = ["sgd_iters", "sinkhorn_iters", "sigma", "lam", "kde_eps", "sinkhorn_eps"]
    pdict = model_config["model_params"]
    del pdict

    # print(data_config.keys())
    output_dir = os.path.join("results",
                              data_config["data_name"],    # e.g. nyu_depth_v2
                              "{}_{}".format(dataset_type, small_run),
                              model_config["model_name"])  # e.g. DORN_nyu_nohints
    safe_makedir(output_dir)

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
         dataset_type,
         save_outputs,
         output_dir,
         data_config,
         seed,
         small_run,
         entry,
         device):
    # Load the model
    model = make_model(**model_config)
    model.eval()
    model.to(device)
    model.sid_obj.to(device)
    model.sinkhorn_opt.to(device)

    # Load the data
    train, test = load_data(dorn_mode=True)
    dataset = test if dataset_type == "test" else train

    from tensorboardX import SummaryWriter
    from datetime import datetime

    init_randomness(seed)

    eval_fn = lambda input_, device: model.evaluate(input_["bgr"].to(device),
                                                    input_["bgr_orig"].to(device),
                                                    input_["crop"][0, :],
                                                    input_["depth_cropped"].to(device),
                                                    torch.ones_like(input_["depth_cropped"]).to(device),
                                                    device)

    if entry is None:
        print("Evaluating the model on {} ({}).".format(data_config["data_name"],
                                                        dataset_type))
        ex.observers.append(FileStorageObserver.create(os.path.join(output_dir, "runs")))
        evaluate_model_on_dataset(eval_fn, dataset, small_run, device, save_outputs, output_dir)
    else:
        print("Evaluating {}".format(entry))
        model.sinkhorn_opt.writer = SummaryWriter(log_dir=os.path.join("runs",
                                                          datetime.now().strftime('%b%d'),
                                                          datetime.now().strftime('%H-%M-%S_') + \
                                                          "dorn_hist_match_wass"))
        evaluate_model_on_data_entry(eval_fn, dataset, entry, device, save_outputs, output_dir)


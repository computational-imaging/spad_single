#!/usr/bin/env python3
import os
import torch
import json
from torch.utils.data import DataLoader
from utils.train_utils import init_randomness, worker_init_randomness
from models.core.checkpoint import load_checkpoint, safe_makedir
from models import make_model
from sacred import Experiment
from time import perf_counter

# Dataset
from models.data.nyuv2_official_hints_sid_dataset import nyuv2_hints_sid_ingredient, load_data

ex = Experiment('eval_hints_sid_bayesian', ingredients=[nyuv2_hints_sid_ingredient])


# Tensorboardx
# writer = SummaryWriter()

@ex.config
def cfg(data_config):
    model_config = {                            # Load pretrained model for testing
        "model_name": "DORN_nyu_hints",
        "model_params": {
            "hints_len": 68,
            "spad_weight": 1.,
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
        "model_state_dict_fn": None
    }
    # ckpt_file = "checkpoints/Mar15/04-10-54_DORN_nyu_hints_nyu_depth_v2/checkpoint_epoch_9_name_fixed.pth.tar"
    ckpt_file = None # Bayesian hints eval
    eval_config = {
        "dataset": "test",                       # {val, test}
        "mode": "save_outputs",                 # {save_outputs, evaluate_metrics}
        "output_dir": "./data/dorn_hints_bayesian_test",
        "entry": None                           # If we want to evaluate on a single entry
    }
    seed = 95290421

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
         eval_config,
         data_config,
         seed,
         device):
    # Load the model
    model = make_model(**model_config)
    model.to(device)
    # model.sid_obj.to(device)
    print(model)

    # Load the data
    _, val, test = load_data()
    dataset = test if eval_config["dataset"] == "test" else val

    init_randomness(seed)

    # Make dataloader
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            worker_init_fn=worker_init_randomness)
    if eval_config["mode"] == "save_outputs":
        print("Evaluating the model on {}".format(eval_config["dataset"]))
        # Run the model on everything and save everything to disk.
        safe_makedir(eval_config["output_dir"])
        with torch.no_grad():
            model.eval()
            # start = perf_counter()
            # total_eval = 0.
            for i, data in enumerate(dataloader):
                print("Evaluating {}".format(data["entry"][0]))
                # start_eval = perf_counter()
                model.write_eval(data,
                                 os.path.join(eval_config["output_dir"],
                                              "{}_out.pt".format(data["entry"][0])),
                                 device)
                # TESTING
                # if i == 99:
                #     break
                # total_eval += perf_counter() - start_eval
            # total = perf_counter() - start
            # print("avg time over 100 iters: {}".format(total/100.))
            # print("single eval: {}".format(total_eval/100.))
        print("Dataset: {} Output dir: {}".format(eval_config["dataset"],
                                                  eval_config["output_dir"]))
    elif eval_config["mode"] == "evaluate_metrics":
        # Load things and call the model's evaluate function on them.
        metrics = model.evaluate_dir(eval_config["output_dir"], device)
        out_file = os.path.join(eval_config["output_dir"], "metrics.json")
        with open(out_file, "w") as f:
            json.dump(metrics, f)
        print("Saved metrics to {}".format(out_file))
    elif eval_config["mode"] == "save_file":
        safe_makedir(eval_config["output_dir"])

    else:
        print("Unrecognized mode: {}".format(eval_config["mode"]))

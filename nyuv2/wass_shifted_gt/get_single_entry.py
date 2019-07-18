#! /usr/bin/env python3
import sys, os
import torch
from models.data.nyuv2_labeled_dataset import cfg, load_data
# Load data
data_config = cfg()
del data_config["data_name"], data_config["bgr_mode"], data_config["channels_first"]
train, test = load_data(**data_config, dorn_mode=True)

entry = int(sys.argv[1])
gt = test[entry]["depth_cropped"]
torch.save(gt, os.path.join("data",
                            "test_{}.pt".format(entry))
           )
# gt = torch.load("test_{}.pt".format(entry))

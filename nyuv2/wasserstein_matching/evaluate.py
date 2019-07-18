#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader
from utils.train_utils import init_randomness
from utils.eval_utils import evaluate_model_on_dataset
from models.core.checkpoint import load_checkpoint, safe_makedir
from models import make_model
from sacred import Experiment
from sacred.observers import FileStorageObserver

import numpy as np
import pandas as pd

# Model
from wasserstein_hist_match import wasserstein_match
from models.data.data_utils.sid_utils import SID
# Dataset

ex = Experiment('eval_dorn_nyuv2_labeled')

@ex.config
def cfg():
    initializer = "dorn"
    entry = None
    save_outputs = True
    small_run = 0
    # Threshold below which we zero out small values in the coupling matrix.
    opt_eps = 1e-4

    # SID Parameters
    sid_bins = 68
    alpha = 0.6569154266167957
    beta = 9.972175646365525,
    offset = 0.

@ex.automain
def run(initializer,
        opt_eps,
        sid_bins,
        alpha,
        beta,
        offset):
    # Load Data
    dataset = np.load(os.path.join("data", initializer, "all_outputs.npy"))

    # Create SID object
    sid_obj = SID(sid_bins, alpha, beta, offset)
    outputs = []
    for i in range(len(dataset)):
        depth_init = dataset[i]
        print(depth_init.shape)





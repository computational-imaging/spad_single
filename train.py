#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

# Load training options from file
import depthnet.model as m
import depthnet.utils as u
import depthnet.data as d
import depthnet.train_utils as tu
from depthnet.options import opt # Reads command line options

if torch.cuda.is_available():
    # Set Device
    torch.cuda.set_device(opt.cuda_device)
    print("Cuda enabled on device: {}".format(torch.cuda.current_device()))

# Tensorboardx
writer = SummaryWriter()
# data_trainloss = "data/trainloss"
# data_valloss = "data/valloss"
# Image = "Image"

# Load data
train, val = d.load_data(trainFile = opt.trainFile,
                         trainDir = opt.trainDir,
                         valFile = opt.valFile,
                         valDir = opt.valDir)

trainLoader = DataLoader(train, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
valLoader = None
if val is not None:
    valLoader = DataLoader(val, batch_size=opt.val_batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Load model, loss, and scheduler
loss = None
if opt.loss == "berhu":
    from depthnet.model import berhu
    loss = berhu
elif opt.loss == "l2":
    loss = MSELoss()
    if torch.cuda.is_available():
        loss.cuda()
elif opt.loss == "l1":
    loss = L1Loss()
    if torch.cuda.is_available():
        loss.cuda()

setup = tu.setup_training(opt, writer)
# Run Training
tu.train(**setup, opt=opt, loss=loss, trainLoader=trainLoader, valLoader=valLoader, test_run=opt.test_run, writer=writer)


writer.close()

#! /bin/bash

python train.py unet data/sunrgbd_all/val.txt data/sunrgbd_all --valFile data/sunrgbd_all/val.txt --valDir data/sunrgbd_all --milestones 25 --num-epochs 50 --cuda-device 2 --test-run

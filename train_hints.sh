#! /bin/bash

python train.py depth_hints data/sunrgbd_nyu/dev.txt data/sunrgbd_nyu --valFile data/sunrgbd_nyu/dev.txt --valDir data/sunrgbd_nyu --milestones 25 --num-epochs 50 --cuda-device 1

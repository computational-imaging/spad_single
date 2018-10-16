#! /bin/bash

nohup python train.py depth data/sunrgbd_nyu/train.txt data/sunrgbd_nyu --valFile data/sunrgbd_nyu/dev.txt --valDir data/sunrgbd_nyu --milestones 25 --num-epochs 50 &



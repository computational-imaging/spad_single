#! /bin/bash

python train.py with no_hints_80 data_config.nyu_depth_v2 cuda_device=\"$1\" -F experiments &
python train.py with hints_80 data_config.nyu_depth_v2 cuda_device=\"$2\" -F experiments &
python train.py with hints_80 data_config.raw_hist data_config.nyu_depth_v2 comment=\"_rawhist\" cuda_device=\"$3\" -F experiments &

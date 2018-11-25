#! /bin/bash
source activate depth-net # conda environment
rmate train.py
rmate depthnet/train_utils.py
rmate depthnet/data.py
rmate depthnet/utils.py

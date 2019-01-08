#! /bin/bash
rmate train.py
rmate test.py
rmate depthnet/train_utils.py
rmate depthnet/data.py
rmate depthnet/utils.py
rmate depthnet/model/wrapper.py
rmate train_test_split.py
rmate depthnet/model/unet_parts.py
rmate depthnet/model/unet_model.py


source activate depth-net # conda environment

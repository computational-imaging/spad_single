# depth-net
Monocular Depth Mapping with ToF Histogram Hints

Inspired by the Global Hints net from Zhang, Zhu et. al. 2017, Real-Time User-Guided Image Colorization with Learned Deep Priors.

Model Architecture:
U-net variant, with a global hints network with 4-5 layers/dilations.

Input to the network:
RGB image
Photon Cloud histogram from SPAD

Output of the network: 
Depth map

This project uses sacred (https://github.com/IDSIA/sacred) to log and save configurations from run to run.

train_test_split_nyu.py - script for splitting the NYU Depth v2 data along the offical train/test split.
depthnet/data.py - functions for loading data
depthnet/train_utils.py - functions for setting up and running the training.
depthnet/utils.py - misc. functions for logging results during the training process.
depthnet/models/loss.py - functions for calculating the loss of the network.
depthnet/models/unet_model.py - functions for building the model.
train.py - the main script for training the network.

Data Directory
data/code_nyu/simulated_data/ConvertRGBD.m - code for extracting the albedo from the RGBD images in the NYU Depth v2 dataset.


Notes:
	- SUNRGBD
		- Raw depth images (i.e. the ones in |depth| folders) are not inpainted. Pixels with missing
		  depth values are assigned a depth value of 0.
		- Inpainted depth images (i.e. the ones in |depth_bfx| folders) are inpainted, but may still
		  have missing depth values. Those values are assigned the same value as the MINIMUM PIXEL VALUE
		  (elsewhere) in the image.



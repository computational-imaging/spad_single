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

train_test_split.py - command line script for extracting stuff from the SUNRGBD directory hierarchy
data.py - functions for manipulating the files and directories created by train_test_split.py
     

Figure out how to simulate SPAD for training.
    - For now, David says to just use regular (normalized?) histograms of depth data.

Notes:
	- SUNRGBD
		- Raw depth images (i.e. the ones in |depth| folders) are not inpainted. Pixels with missing
		  depth values are assigned a depth value of 0.
		- Inpainted depth images (i.e. the ones in |depth_bfx| folders) are inpainted, but may still
		  have missing depth values. Those values are assigned the same value as the MINIMUM PIXEL VALUE
		  (elsewhere) in the image.


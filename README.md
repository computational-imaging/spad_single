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

TODO:
Download NYU Depth data. DONE
Make U-net for NYU Depth.
     - First try to replicate Zhang,Zhu et. al. architecture DONE ish
     

Figure out how to simulate SPAD for training.
     - For now, David says to just use regular (normalized?) histograms of depth data.

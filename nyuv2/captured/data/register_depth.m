kinect_file = "8_29_kitchen_scene/kinect.mat";
load(kinect_file)

%% Perform alignment according to NYU Depth v2
camera_params;

depth_im_swapped = double(swapbytes(depth_im));
[depthOut, rgbOut] = project_depth_map(depth_im_swapped, rgb_im);

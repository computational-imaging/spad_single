#!/usr/bin/env python3
# This script requires the pyrealsense2 wrapper for librealsense.
# For installation instructions, see:
# Windows/Linux:
# https://github.com/IntelRealSense/librealsense
# Mac:
# https://github.com/GitwellAnyohub/Real_Sensible_Moseq/blob/master/bash_and_install/install_realsense_sdk_mac_os.sh

import sys, os
sys.path.append("/usr/local/lib")   # Necessary for pyrealsense2 import for my install on mac
import pyrealsense2 as rs
import numpy as np
import cv2
from argparse import ArgumentParser

parser = ArgumentParser(description="Capturing data from the realsense")
parser.add_argument("--output-dir", default="testing", help="The directory to save the captured images to.")
args = parser.parse_args()

def safe_makedir(path):
    """Makes a directory, or returns if the directory
    already exists.

    Taken from:
    https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

safe_makedir(args.output_dir)

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 10.  # 10 meters
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
i = 0   # Indexes captured images
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x480 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to black
        # Convention for NYU Depth v2
        black_color = 0.
        depth_image[depth_image > clipping_distance] = black_color

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(32)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        elif key & 0xFF == ord(']'):
            # Capture current image
            # Convert depth image to png scale
            depth_image_u16 = (depth_image.astype('float')*(2**16-1)/clipping_distance).astype('uint16')
            cv2.imwrite(os.path.join(args.output_dir, "{}_rgb.png".format(i)), color_image)
            cv2.imwrite(os.path.join(args.output_dir, "{}_rawDepth.png".format(i)), depth_image_u16)
            i += 1
finally:
    pipeline.stop()

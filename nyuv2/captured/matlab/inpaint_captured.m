% Inpainting the small realsense dataset
% Parameters
addpath('nyu_utils');
rootdir = "../figures";
models = {'midas'}
scenes = {"8_29_kitchen_scene", "8_29_conference_room_scene", "8_30_conference_room2_scene", "8_30_Hallway", ...
          "8_30_poster_scene", "8_30_small_lab_scene", "8_31_outdoor3"}
n_images = [4, 3, 3, 4, 4, 4, 5];
%maxDepth = 10;

% Inpaint and save
for m = 1:length(models)
    for s = 1:length(scenes)
        disp(join(["Inpainting ", models{m}, "[", scenes{s}, "]"], ""))
        sceneDir = join([rootdir, models{m}, scenes{s}], "/");
        % Load Images
        rgbFile = join([sceneDir, "rgb_cropped.png"], "/");
        rgb = double(imread(rgbFile));
        depthFile = join([sceneDir, "gt_z_proj_crop.png"], "/");
        rawDepth = double(imread(depthFile));
        % Inpaint depth using depth and rgb
    %     rawDepthCropped = crop_image(rawDepth);
        depthFilled = fill_depth_colorization(rgb, rawDepth);
%        depthFilled(depthFilled > maxDepth) = maxDepth;

        % Save images
        depthFilledFile = join([sceneDir, "gt_z_proj_crop_filled.png"], "/");
        depthFilledImg = uint16(depthFilled);
        imwrite(depthFilledImg, depthFilledFile)
    end
end



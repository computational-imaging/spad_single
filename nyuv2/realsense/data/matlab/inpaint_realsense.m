% Inpainting the small realsense dataset
% Parameters
addpath('nyu_utils');
rootdir = "..";
datasetDirs = {'conf_room', 'couches', 'kitchen', 'kitchen2', 'office', 'office2', 'third_floor'};
n_images = [4, 3, 3, 4, 4, 4, 5];
maxDepth = 10;

% Inpaint and save
for ii = 1:length(datasetDirs)
    for i = 0:n_images(ii)-1
        disp(join(["Inpainting ", datasetDirs{ii}, "[", i, "]"], ""))
        datasetDir = join([rootdir, datasetDirs{ii}], "/");
        % Load Images
        rgbFile = join([datasetDir, "/", i, "_rgb.png"], "");
        rgb = double(imread(rgbFile));
        depthFile = join([datasetDir, "/", i, "_rawDepth.png"], "");
        rawDepth = double(imread(depthFile));
        rawDepth = rawDepth * maxDepth/(2^16-1);
        % Inpaint depth using depth and rgb
    %     rawDepthCropped = crop_image(rawDepth);
        depthFilled = fill_depth_colorization(rgb, rawDepth);
        depthFilled(depthFilled > maxDepth) = maxDepth;

        % Save images
        depthFilledFile = join([datasetDir, "/", i, "_depth.png"], "");
        depthFilledImg = uint16(depthFilled*(2^16-1)/maxDepth);
        imwrite(depthFilledImg, depthFilledFile)
    end
end



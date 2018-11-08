% Auxilary scripts
addpath(genpath('./intrinsic_texture'));
addpath('nyu_utils');

% Directory of raw depth info
datasetDir = '../../sunrgbd_all/';

% get the scene names
% scenes = ls(datasetDir);
% scenes = regexp(scenes, '(\s+|\n)', 'split');
% scenes(end) = [];

% Read info on the dataset.
infoFile = [datasetDir 'info.json'];
info = jsondecode(fileread(infoFile));
infoNames = fieldnames(info);
num_images = getfield(info, 'num_images');
infoNames = infoNames(1:end-1); % Remove 'num_images' at end
imageIDs = cell(1, num_images);
for i = 1:size(infoNames)
    name = infoNames(i);
%     disp(name)
%     if i / 5==1
%         break
%     end
%     disp(name{1}(2:end));
%     disp(name(2:end))
    imageIDs{i} = {name{1}(2:end)};
end
disp(imageIDs)
% % e.g. infoNames is now {'x0'}, {'x1'}, ...
%% 

% Initialize camera parameters
camera_params;

p = parpool('SpmdEnabled', false);
parfor i = 1:num_images
    try
        imageID = imageIDs{i};
        rawDepthFile = strcat(datasetDir, imageID, '_rawdepth.png');
        rawDepthFile = rawDepthFile{1};
        depthFile = strcat(datasetDir, imageID, '_depth.png');
        depthFile = depthFile{1};
        rgbFile = strcat(datasetDir, imageID, '_rgb.png');
        rgbFile = rgbFile{1};

        disp('starting!');

        % The name of the scene to demo.
    %     outdir = ['./processed/' sceneName];
    %     mkdir(outdir);
        albedo_out = strcat(datasetDir, imageID, '_albedo.png');
        albedo_out = albedo_out{1};

        compare_depth_out = strcat(datasetDir, imageID, '_depth_comp.png');
        compare_depth_out = compare_depth_out{1};

    % The absolute directory of the 
%     sceneDir = sprintf('%s/%s', datasetDir, sceneName);

    % Reads the list of frames.
%     frameList = get_synched_frames(sceneDir);

    % Displays each pair of synchronized RGB and Depth frames.
%     idx = 1 : 10 : numel(frameList);
    
%     for ii = 1:length(idx)
        % check if already exists
%         depth_out = sprintf('%s/depth_%04d.mat', outdir, idx(ii));
%         intensity_out = sprintf('%s/intensity_%04d.mat', outdir, idx(ii));
%         dist_out = sprintf('%s/dist_%04d.mat',outdir, idx(ii));
%         dist_out_hr = sprintf('%s/dist_hr_%04d.mat',outdir, idx(ii));



%     try
%         imgRgb = imread([sceneDir '/' frameList(idx(ii)).rawRgbFilename]);
        imgRgb = imread(rgbFile);
%         imgDepthRaw = swapbytes(imread([sceneDir '/' frameList(idx(ii)).rawDepthFilename]));
        imgDepthAbs = imread(rawDepthFile);

        % Crop the images to include the areas where we have depth information.
%         imgRgb = crop_image(imgRgb);
%         imgDepthProj = project_depth_map(imgDepthRaw, imgRgb);
%         imgDepthAbs = crop_image(imgDepthProj);
        imgDepthFilled = fill_depth_cross_bf(imgRgb, double(imgDepthAbs));
%         imgDepthFilledCompare = imread(depthFile); % Testing my suspicions
        imwrite(imgDepthFilled, compare_depth_out); % Testing my suspicions

        if exist(albedo_out,'file')
            disp('continuing');
            continue;
        end
        
        H = size(imgDepthFilled, 1);
        W = size(imgDepthFilled, 2);
        % get distance from the depth image
        cx = cx_d - 41 + 1;
        cy = cy_d - 45 + 1;
        [xx,yy] = meshgrid(1:W, 1:H);
        X = (xx - cx) .* imgDepthFilled / fx_d;
        Y = (yy - cy) .* imgDepthFilled / fy_d;
        Z = imgDepthFilled;
        imgDist_hr = sqrt(X.^2 + Y.^2 + Z.^2);

        % estimate the albedo image and save the outputs
        I = im2double(imgRgb);
%         I = imresize(I, [512, 512], 'bilinear');
%         imgDepthFilled = imresize(imgDepthFilled, [512,512], 'bilinear');
%         imgDist = imresize(imgDist_hr, [256,256], 'bilinear');
%         imgDist_hr = imresize(imgDist_hr, [512,512], 'bilinear');
        S = RollingGuidanceFilter(I, 3, 0.1, 4);
        [albedo, ~] = intrinsic_decomp(I, S, imgDepthFilled, 0.0001, 0.8, 0.5);
%         intensity = rgb2gray(I);
% 
%         dist = imgDist;
%         intensity = im2uint8(intensity);
%         dist_hr = imgDist_hr;
%         ConvertRGBDParsave(albedo_out, dist_out, intensity_out, dist_out_hr, albedo, dist, intensity, dist_hr)
        imwrite(albedo, albedo_out);
        disp('done with this one.')
    catch e
        fprintf(1,'ERROR: %s\n',e.identifier);
        fprintf(1,'%s',e.message);
        fprintf(1,'%s',rgbFile);
        continue;
    end
%     break; % For testing
end

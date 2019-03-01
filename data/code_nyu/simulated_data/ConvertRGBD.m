% The directory where you extracted the raw dataset.
addpath(genpath('./intrinsic_texture'));
addpath('nyu_utils');
datasetDir = '../../nyu_depth_v2';

% get the scene names
scenes = ls(datasetDir);
scenes = regexp(scenes, '(\s+|\n)', 'split');
scenes(end) = [];

addpath(genpath('./intrinsic_texture'));
addpath('./nyu_utils');
camera_params; % Defines maxDepth

% TESTING
% disp(scenes)
% scenes = scenes(1);
% exit;

p = parpool(28)
parfor ss = 1:length(scenes)
    sceneName = scenes{ss};

    disp('starting!');

    % The name of the scene to demo.
    outdir = ['../../nyu_depth_v2_scaled16/' sceneName];
    mkdir(outdir);

    % The absolute directory of the 
    sceneDir = sprintf('%s/%s', datasetDir, sceneName);

    % Reads the list of frames.
    frameList = get_synched_frames(sceneDir);

    % Displays each pair of synchronized RGB and Depth frames.
    % Take one out of every 10 frames.
    idx = 1 : numel(frameList);
    % idx = 1 : 100 : numel(frameList);

    for ii = 1:length(idx)
        % check if already exists
        rgb_out = sprintf('%s/%04d_rgb.png', outdir, idx(ii))
        rawdepth_out = sprintf('%s/%04d_rawdepth.png', outdir, idx(ii))
        depth_out = sprintf('%s/%04d_depth.png', outdir, idx(ii));
        albedo_out = sprintf('%s/%04d_albedo.png', outdir, idx(ii));
        % intensity_out = sprintf('%s/intensity_%04d.mat', outdir, idx(ii));
        % dist_out = sprintf('%s/dist_%04d.mat',outdir, idx(ii));
        % dist_out_hr = sprintf('%s/dist_hr_%04d.mat',outdir, idx(ii));

        % rawDepthFile = strcat(outdir, imageID, '_rawdepth.png');
        % rawDepthFile = rawDepthFile{1};
        % depthFile = strcat(outidr, imageID, '_depth.png');
        % depthFile = depthFile{1};
        % rgbFile = strcat(outdir, imageID, '_rgb.png');
        % rgbFile = rgbFile{1};

        if exist(rgb_out, 'file') && exist(rawdepth_out,'file') ...
                && exist(depth_out, 'file') && exist(albedo_out,'file')
            disp('continuing');
            continue;
        end

        
        try
            imgRgb = imread([sceneDir '/' frameList(idx(ii)).rawRgbFilename]);
            imgDepthRaw = swapbytes(imread([sceneDir '/' frameList(idx(ii)).rawDepthFilename]));

            % Crop the images to include the areas where we have depth information.
            imgRgb = crop_image(imgRgb);
            imgDepthProj = project_depth_map(imgDepthRaw, imgRgb);
            imgDepthAbs = crop_image(imgDepthProj);
            imgDepthFilled = fill_depth_cross_bf(imgRgb, double(imgDepthAbs));
          
            % get distance from the depth image
            % cx = cx_d - 41 + 1;
            % cy = cy_d - 45 + 1;
            % [xx,yy] = meshgrid(1:561, 1:427);
            % X = (xx - cx) .* imgDepthFilled / fx_d;
            % Y = (yy - cy) .* imgDepthFilled / fy_d;
            % Z = imgDepthFilled;
            % imgDist_hr = sqrt(X.^2 + Y.^2 + Z.^2);
           
            % estimate the albedo image and save the outputs
            I = im2double(imgRgb);
            % I = imresize(I, [512, 512], 'bilinear');
            % imgDepthFilled = imresize(imgDepthFilled, [512,512], 'bilinear');
            % imgDist = imresize(imgDist_hr, [256,256], 'bilinear');
            % imgDist_hr = imresize(imgDist_hr, [512,512], 'bilinear');
            S = RollingGuidanceFilter(I, 3, 0.1, 4);
            [albedo, ~] = intrinsic_decomp(I, S, imgDepthFilled, 0.0001, 0.8, 0.5);
            if albedo == -1
                continue;
            end
            % Save rgb (I), rawdepth (imgDepthAbs), depth (imgDepthFilled), and albedo (albedo)
            % (all at the same resolution)
            imwrite(I, rgb_out);
            % Rescale depth to use entire range of image bits
            % 16 bits - max value is 2^16-1
            % maxDepth (defined in camera_params) = 10 (for nyu_v2)
            imwrite(uint16(imgDepthAbs*(2^16-1)/maxDepth), rawdepth_out);
            imwrite(uint16(imgDepthFilled*(2^16-1)/maxDepth), depth_out);
            imwrite(albedo, albedo_out);

            % intensity = rgb2gray(I);

            % dist = imgDist;
            % intensity = im2uint8(intensity);
            % dist_hr = imgDist_hr;
            % ConvertRGBDParsave(albedo_out, dist_out, intensity_out, dist_out_hr, albedo, dist, intensity, dist_hr)
             
        catch e
            fprintf(1,'ERROR: %s\n',e.identifier);
            fprintf(1,'%s',e.message);
            continue;
        end
    end
end
exit;

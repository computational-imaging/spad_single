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
% disp(imageIDs)
% % e.g. infoNames is now {'x0'}, {'x1'}, ...
%% 

% Initialize camera parameters
camera_params;

p = parpool(28);
parfor i = 1:num_images
    try
        t = getCurrentTask();
        workerid = t.ID;
        % disp(class(workerid))
        % logfile = strcat(num2str(workerid), '_log.txt');
        blacklistfile = strcat(num2str(workerid), '_blacklist.txt');
        % Clear Blacklist
        if isfile(blacklistfile)
            delete(blacklistfile);
        end
        % fileID = fopen(logfile, 'a');

        imageID = imageIDs{i};
        % imageID = {'4597'};
        rawDepthFile = strcat(datasetDir, imageID, '_rawdepth.png');
        rawDepthFile = rawDepthFile{1};
        depthFile = strcat(datasetDir, imageID, '_depth.png');
        depthFile = depthFile{1};
        rgbFile = strcat(datasetDir, imageID, '_rgb.png');
        rgbFile = rgbFile{1};

        disp('starting!');
        % fprintf(fileID, strcat('starting:', imageID{1}, '\n'));

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
        % disp(class(imgDepthAbs))
        imgDepthConverted = bitor(bitshift(imgDepthAbs, 16-3) , bitshift(imgDepthAbs, -3));
        imgDepthTrue = double(imgDepthConverted)/1000;
        imgDepthTrue(imgDepthTrue > 8) = 8;
        % Crop the images to include the areas where we have depth information.
    %         imgRgb = crop_image(imgRgb);
    %         imgDepthProj = project_depth_map(imgDepthRaw, imgRgb);
    %         imgDepthAbs = crop_image(imgDepthProj);
        imgDepthFilled = fill_depth_cross_bf(imgRgb, imgDepthTrue);

    %         imgDepthFilledCompare = imread(depthFile); % Testing my suspicions
        % disp(class(imgDepthFilled))
        % imgDepthOut = imgDepthFilled/8;
        imgDepthOut = uint16(round(imgDepthFilled*(65535/8)));
        imwrite(imgDepthOut, compare_depth_out); % Testing my suspicions

        if exist(albedo_out,'file')
            disp('continuing');
            % fprintf(fileID, strcat('finished:', imageID{1}, '\n'));
            % fclose(fileID);
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
        % I = imresize(I, [440, 590], 'bilinear');
        % I = imresize(I, [256, 256], 'bilinear');
        % imgDepthFilled = imresize(imgDepthFilled, [440, 590], 'bilinear');
    %         imgDist = imresize(imgDist_hr, [256,256], 'bilinear');
    %         imgDist_hr = imresize(imgDist_hr, [512,512], 'bilinear');
        S = RollingGuidanceFilter(I, 3, 0.1, 4);
        [albedo, ~] = intrinsic_decomp(I, S, imgDepthFilled, 0.0001, 0.8, 0.5);
        if albedo == -1
            f = fopen(blacklistfile, "a");
            fprintf(f, strcat(imageID{1}, '\n'));
            fclose(f);
            disp(strcat("blacklisted: ", imageID{1}));
            continue;
        end
    %         intensity = rgb2gray(I);
    % 
    %         dist = imgDist;
    %         intensity = im2uint8(intensity);
    %         dist_hr = imgDist_hr;
    %         ConvertRGBDParsave(albedo_out, dist_out, intensity_out, dist_out_hr, albedo, dist, intensity, dist_hr)
        imwrite(albedo, albedo_out);
        disp('done with this one.')
        % fprintf(fileID, strcat('finished:', imageID{1}, '\n'));
        % fclose(fileID);
    catch e
        fprintf(2,'ERROR: %s\n',e.identifier);
        fprintf(2,'%s',e.message);
        % fclose(fileID);
    end
    %     break; % For testing
end

clear all; close all;
files = {"test_int_True_fall_True_dc_100000.0_jit_False_poiss_True_spad.mat", ...
         "test_int_True_fall_True_dc_20000.0_jit_False_poiss_True_spad.mat", ...
         "test_int_True_fall_True_dc_10000.0_jit_False_poiss_True_spad.mat"};

for i=1:length(files)
    file = files{i};
    data = load(file);
    spad = double(data.data);
    spad_denoised = wdenoise(spad', 10, 'DenoisingMethod', 'BlockJS')';
    newFile = join([extractBefore(files{i}, ".mat"), "_denoised.mat"],"");
    save(newFile, 'spad_denoised')
end

% Diagnostic
figure;
bar(spad(1,1:100)),set(gca, 'yscale', 'log');
figure;
bar(spad_denoised(1,1:100)),set(gca, 'yscale', 'log');

figure;
bar(spad(1,250:350)),set(gca, 'yscale', 'log');
figure;
bar(spad_denoised(1, 250:350)),set(gca, 'yscale', 'log');

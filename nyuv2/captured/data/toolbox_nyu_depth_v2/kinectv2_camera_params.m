% Intrinsic parameters taken from Kim et. al 2015.
%   
% Extrinsic parameters taken from factory row for Basso et. al. 2018.
% 

% The maximum depth used, in meters.
maxDepth = 10;

% RGB Intrinsic Parameters
fx_rgb = 5.1885790117450188e+02;
fy_rgb = 5.1946961112127485e+02;
cx_rgb = 3.2558244941119034e+02;
cy_rgb = 2.5373616633400465e+02;

% RGB Distortion Parameters
k1_rgb =  2.0796615318809061e-01;
k2_rgb = -5.8613825163911781e-01;
p1_rgb = 7.2231363135888329e-04;
p2_rgb = 1.0479627195765181e-03;
k3_rgb = 4.9856986684705107e-01;

% Depth Intrinsic Parameters
fx_d = 5.8262448167737955e+02;
fy_d = 5.8269103270988637e+02;
cx_d = 3.1304475870804731e+02;
cy_d = 2.3844389626620386e+02;

% RGB Distortion Parameters
k1_d = -9.9897236553084481e-02;
k2_d = 3.9065324602765344e-01;
p1_d = 1.9290592870229277e-03;
p2_d = -1.9422022475975055e-03;
k3_d = -5.1031725053400578e-01;

% Rotation
R = eye(3);

% 3D Translation
t_x = 0.052;
t_z = 0.0;
t_y = 0.0;

% Parameters for making depth absolute.
depthParam1 = 351.3;
depthParam2 = 1092.5;

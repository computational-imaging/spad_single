load('diffuser_10s.mat');
load('scanned_10s_512res.mat');

scanned = scanned{1}; % grab first 10 s exposure
scanned = sum(sum(scanned, 2), 3); % sum over spatial dims


% sum the 10 x 1 s exposures to get total of 10 s exposure time
tmp = diffuser{1};
for ii = 2:10
   tmp = tmp + diffuser{ii};
end
diffuser = tmp;
diffuser = single(diffuser); % convert to single precision

% diffuser is 4 ps binning, but scanned is 16 ps
% average over 4 bins to convert to 16 ps
diffuser = diffuser(1:2:end) + diffuser(2:2:end);
diffuser = diffuser(1:2:end) + diffuser(2:2:end);

% align the peaks
% note that the scanned measurement has the correct absolute distance
% because it is calibrated.
% the diffuser measurement was captured with the free-running spad, 
% and there's an unknown timing offset. So we're just going to align it
% with the captured scanned measurement.
[~, diffuser_peak] = max(diffuser);
[~, scanned_peak] = max(scanned);
offset = diffuser_peak - scanned_peak;
diffuser = circshift(diffuser, -offset);

% truncate to be the same size
diffuser = diffuser(1:2048);
scanned = scanned(1:2048);

diffuser_norm = diffuser / max(diffuser);
scanned_norm = scanned / max(scanned);

t = (0:2047) * 16e-12 * 1e9;
plot(t, scanned_norm, 'linewidth', 2);
hold on; 
plot(t, diffuser_norm, 'linewidth', 2);
hold off;
xlabel('time (ns)');
ylabel('normalized photon count');
grid on;
set(gcf, 'color', 'white');
legend('scanned', 'diffused');

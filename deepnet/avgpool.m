% convnet_avgpool
% Copyright (C) 2013 KyungHyun Cho
%
%This program is free software; you can redistribute it and/or
%modify it under the terms of the GNU General Public License
%as published by the Free Software Foundation; either version 2
%of the License, or (at your option) any later version.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program; if not, write to the Free Software
%Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%

% TODO: replace it with a more efficient routine.
function [out, outmap] = avgpool (resp, ratios)

% % % try
% % %    use_gpu = gpuDeviceCount;
% % % catch errgpu
% % %    use_gpu = false;
% % %    disp(['Could not use CUDA. Error: ' errgpu.identifier])
% % % end

% % % use_gpu = 0;

subwindow_avg = 1 / (ratios(1) * ratios(2)) * ones(ratios);
subavg = convn(resp, reshape(subwindow_avg, [1, ratios(1), ratios(2), 1]), 'valid');
subavg = subavg(:, 1:ratios(1):end, :);
subavg = subavg(:, :, 1:ratios(2):end);

out = subavg;
outmap = ones(size(resp)); % kind of waste..






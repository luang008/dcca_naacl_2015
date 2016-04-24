% convnet_maxpool
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
function [out, outmap] = maxpool (in, ratios)

% % % try
% % %   use_gpu = gpuDeviceCount;
% % % catch errgpu
% % %   use_gpu = false;
% % %   disp(['Could not use CUDA. Error: ' errgpu.identifier])
% % % end

% % % use_gpu = 0;

% because of different styles of indexing, some permutation is required.
% to use outidx, one needs to permute the original images in a batch and
% use the outidx as a linear index

% % % if use_gpu
% % %     outmap = parallel.gpu.GPUArray.zeros(size(in));
% % %     [out, outidx] = MaxPooling(permute(gather(in),[2, 3, 1]), single(ratios));
% % %     outmap = permute(outmap, [2, 3, 1]);
% % %     outmap(outidx) = 1;
% % %     outmap = permute(outmap, [3, 1, 2]);
% % %     out = gpuArray(permute(out, [3, 1, 2]));
% % % else
outmap = zeros(size(in));
[out, outidx] = MaxPooling(permute(double(gather(in)),[2, 3, 1]), ratios);
outmap = permute(outmap, [2, 3, 1]);
outmap(outidx) = 1;
outmap = permute(outmap, [3, 1, 2]);
out = permute(out, [3, 1, 2]);
% % % end

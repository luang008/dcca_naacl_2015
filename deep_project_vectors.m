% Generate projected vectors using Deep CCA


function deep_project_vectors(H1,H2,rcov1,rcov2,batchsize,eta0,momentum)

addpath ./deepnet/;

% Perform Deep CCA on the subset of the aligned vectors
filename = testDCCA(H1,H2,rcov1,rcov2,batchsize,eta0,momentum);

load orig_data origEnVecs
% Normalize the matrices by rows. Because subset was normalized.
origEnVecs = normr(origEnVecs); 

load(filename,'F1opt','mean1','s1'); 
% Go though the same data normalization as training data in DCCA.
origEnVecs = bsxfun(@times, bsxfun(@minus,origEnVecs,mean1), 1./s1);

% Apply learned DCCA mapping.
origEnVecsProjected = deepnetfwd_big(origEnVecs,F1opt);
clear origEnVecs;

% Zero mean and scale each row to unit length.
origEnVecsProjected = bsxfun(@minus, origEnVecsProjected, mean(origEnVecsProjected));
origEnVecsProjected = normr(origEnVecsProjected);

% Write the projected english vectors to file
filename = filename(1:(length(filename)-4));
dlmwrite([filename '.txt'], origEnVecsProjected, ' ');

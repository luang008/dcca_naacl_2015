function alltask(filename,H1)
addpath ./deepnet/;

load([filename '.mat'],'F1opt','F2opt','mean1','mean2','s1','s2','CORR2'); 
load orig_data;

% Normalize the matrices by rows. Because subset was normalized.
origEnVecs = normr(origEnVecs); 
origForeignVecs = normr(origForeignVecs);

% Go though the same data normalization as training data in DCCA.
origEnVecs = bsxfun(@times, bsxfun(@minus,origEnVecs,mean1), 1./s1);
origForeignVecs = bsxfun(@times,bsxfun(@minus,origForeignVecs,mean2),1./s2);

% Apply learned DCCA mapping.
origEnVecsProjected = deepnetfwd_big(origEnVecs,F1opt);
origForeignVecsProjected = deepnetfwd_big(origForeignVecs,F2opt);
clear origEnVecs;
clear origForeignVecs;

% Zero mean and scale each row to unit length.
origEnVecsProjected = bsxfun(@minus, origEnVecsProjected, mean(origEnVecsProjected));
origEnVecsProjected = normr(origEnVecsProjected);
origForeignVecsProjected = bsxfun(@minus, origForeignVecsProjected, mean(origForeignVecsProjected));
origForeignVecsProjected = normr(origForeignVecsProjected);

% Write the projected english vectors to file
dlmwrite([filename '.txt'], origEnVecsProjected, ' ');

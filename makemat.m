function y = makemat(origForeignVecFile,origEnVecFile,subsetEnVecFile, subsetForeignVecFile)

% first column is words, hence not being read
origEnVecs = dlmread(origEnVecFile, ' ', 0, 1);
origForeignVecs = dlmread(origForeignVecFile, ' ', 0, 1);
subsetEnVecs = dlmread(subsetEnVecFile, ' ', 0, 1);
subsetForeignVecs = dlmread(subsetForeignVecFile, ' ', 0, 1);

save subset_data subsetEnVecs subsetForeignVecs
save orig_data origEnVecs origForeignVecs
% Delete all matrices from memory
clear;

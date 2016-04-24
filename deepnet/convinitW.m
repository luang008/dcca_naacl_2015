function f=convinitW(f)
% initialize the weights of a convolutional layer.

convdin=f.filternumrows*f.filternumcols*f.numinputmaps;
convdout=f.numoutputmaps;
W=2*(rand(convdin,convdout)-0.5)/sqrt(convdin);
f.W=[W;rand(1,f.numoutputmaps)*0.01];

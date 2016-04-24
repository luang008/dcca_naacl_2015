function filename = testDCCA(H1,H2,rcov1,rcov2,batchsize,eta0,momentum)
  K = H1(length(H1))
% Hidden activation type.
hiddentype='relu';
% Final CCA dimension.
% Architecture (hidden layer sizes) for view 1 neural network.
NN1=H1;
% Architecture (hidden layer sizes)  for view 2 neural network.
NN2=H2;
% Regularizations for each view: IMPORTANT, SHOULD BE TUNED.
%rcov1=rcov1;
%rcov2=rcov2;
% Weight decay parameter, 5e-4 is reasonable.
l2penalty=0.0005;
% Minibatchsize: IMPORTANT, SHOULD BE TUNED.
%batchsize=3000;
% Learning rate: IMPORTANT, SHOULD BE TUNED.
%eta0=0.005;
% Rate in which learning rate decays over iterations.
% 1 means constant learning rate: this is a reasonable value.
decay=1;
% Momentum: IMPORTANT, SHOULD BE TUNED, AROUND 0.9 is TYPICALLY GOOD.
%momentum=0.9;
% Dropout probability, set it to zero for now.
dropprob=0;
% How many passes of the data you run SGD with.
maxepoch=2;
% Add the path where you put all these matlab files.
% addpath /share/data/speech-multiview/wwang5/cca/;
% Run the demo.
demo_SGD;


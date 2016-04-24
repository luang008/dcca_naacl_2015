
%% Path to your input data. Adapt it to your setting.

%% Add the deepnet path so that you can call the auxiliary functions.
addpath deepnet/;

%% Filename to save your and results.
strh1 = '';
strh1 = [strh1 num2str(H1(1))];
      strh2 = '';
strh2 =[strh2 num2str(H2(1))];
for i = (2:length(H1))
  strh1 = [strh1 ',' num2str(H1(i))];
end
for i = (2:length(H2))
  strh2 = [strh2 ',' num2str(H2(i))];
end
%% Adapt it to your problem.
filename=['result_' 'H1=[' strh1 ']_H2=[' strh2 ...
  ']_rcov1=' num2str(rcov1) '_rcov2=' num2str(rcov2) ...
  '_batchsize=' num2str(batchsize) ...
  '_eta0=' num2str(eta0) ...
  '_momentum=' num2str(momentum) ...
  '.mat'];

rcov=[rcov1 rcov2];

%% Load your data files.
%% Your training data is called X1 and X2 for view 1 and view 2 respectively.
%% Your tuning data is called XV1 and XV2 for view 1 and view 2 respectively.
%% Your testing data is called XTe1 and XTe2 for view 1 and view 2 respectively.
%% YOur data matrix should be NxD, each row of the matrices contain a sample.
load subset_data;
X1=subsetEnVecs(1:33000,:);
X2=subsetForeignVecs(1:33000,:);
XV1=subsetEnVecs(33001:end,:);
XV2=subsetForeignVecs(33001:end,:);
clear subsetEnVecs subsetForeignVecs;

%% Data normalization.
mean1=mean(X1); s1=std(X1);
X1=(X1-repmat(mean1,size(X1,1),1))*diag(1./sparse(s1));
XV1=(XV1-repmat(mean1,size(XV1,1),1))*diag(1./sparse(s1));
mean2=mean(X2); s2=std(X2);
X2=(X2-repmat(mean2,size(X2,1),1))*diag(1./sparse(s2));
XV2=(XV2-repmat(mean2,size(XV2,1),1))*diag(1./sparse(s2));

%% linear CCA performance.
[A,B,~,~,corr1]=linCCA(X1,X2,K,rcov);
corr1=sum(corr1)
corr2=DCCA_corr(XV1*A,XV2*B,K)

%% Run deep CCA.
[N,D1] = size(X1);
[~,D2] = size(X2);

randseed = rng('shuffle');

%% Set the architecture.
Layersizes1 = [D1 NN1];
Layertypes1 = {};
for nn1=1:length(NN1);
  Layertypes1 = [Layertypes1, {hiddentype}];
end
%% I set my last layer to be linear, remove the following line if you do not want it.
if length(Layertypes1)>0 Layertypes1{end}='linear'; end

Layersizes2 = [D2 NN2];
Layertypes2 = {};
for nn2=1:length(NN2);
  Layertypes2 = [Layertypes2, {hiddentype}];
end
%% I set my last layer to be linear, remove the following line if you do not want it.
if length(Layertypes2)>0 Layertypes2{end}='linear'; end

%% Initialize the weights of each layer.
F1_init = deepnetinit(Layersizes1,Layertypes1);
F2_init = deepnetinit(Layersizes2,Layertypes2);
F1 = F1_init; F2 = F2_init;

%% L2 penalty on weights is used for DCCA training.
for j=1:length(F1)
  F1{j}.l = l2penalty;
end
for j=1:length(F2)
  F2{j}.l = l2penalty;
end

%% The last linear CCA step and fix the sign.
FX1=deepnetfwd_big(X1,F1);
FX2=deepnetfwd_big(X2,F2);
[A,B,m1,m2,D]=linCCA(FX1,FX2,K,rcov);
SIGN=sign(A(1,:)+eps);
A=A*diag(sparse(SIGN)); B=B*diag(sparse(SIGN));
A=full(A); B=full(B);
f1.type='linear'; f1.units=K; f1.W=[A;-m1*A]; F1tmp=F1; F1tmp{end+1}=f1;
f2.type='linear'; f2.units=K; f2.W=[B;-m2*B]; F2tmp=F2; F2tmp{end+1}=f2;

%% Feed-forward the data to obtain final layer output.
X_train=deepnetfwd_big(X1,F1tmp);
X_tune=deepnetfwd_big(XV1,F1tmp);
%% Evaluate the initial correlations.
CORR1 = DCCA_corr(X_train,deepnetfwd_big(X2,F2tmp),K);
CORR2 = DCCA_corr(X_tune,deepnetfwd_big(XV2,F2tmp),K);
its=0; TT=0;
optvalid=CORR2;
F1opt=F1tmp; F2opt=F2tmp;
delta=0;
save(filename,'randseed','F1','F2','F1opt','F2opt','TT','delta','optvalid','CORR1','CORR2','mean1','s1','mean2','s2');

%% Set dropout probability.
dropprob1=[dropprob ones(1,length(F1)-1)*dropprob 0]
dropprob2=[dropprob*0 ones(1,length(F2)-1)*0 0]

%% Run stochastic gradient descent.
DCCA_SGD;

save(filename,'randseed','F1opt','F2opt','F1','F2','CORR1','CORR2','delta','optvalid','TT','mean1','s1','mean2','s2');




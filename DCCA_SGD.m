
%% Put all weight parameters in a long vector.
VV = [];
Nlayers = length(F1); net1 = cell(1,Nlayers);
for k=1:Nlayers
  VV = [VV; F1{k}.W(:)];
  net1{k} = rmfield(F1{k},'W'); % net contains information other than weights.
end
Nlayers = length(F2); net2 = cell(1,Nlayers);
for k=1:Nlayers
  VV = [VV; F2{k}.W(:)];
  net2{k} = rmfield(F2{k},'W'); % net contains information other than weights.
end
fprintf('Number of weight parameters: %d\n',length(VV));

tol=0;
numbatches=ceil(N/batchsize);

while its<maxepoch
  
  eta = eta0*decay^its; %% Reduce learning rate if needed.
  
  t0 = tic;
  rp = randperm(N);   %% Shuffle the data set.
  
  for i = 1:numbatches
    % % %     fprintf('minibatch %d\n',i);
    idx1 = (i-1)*batchsize+1;
    idx2 = min(i*batchsize,N);
    idx = [rp(idx1:idx2),rp(1:max(0,i*batchsize-N))];
    X1batch = X1(idx,:);
    X2batch = X2(idx,:);
    
    %% Stochastic gradient computed on a minibatch.
    [E,grad] = DCCA_convgrad(VV,X1batch,X2batch,net1,net2,K,rcov,dropprob1,dropprob2);
    %% Gradient descent with momentum.
    delta = momentum*delta - eta*grad;
    VV = VV + delta;
  end
  
  %% Record the time spent for each epoch.
  its=its+1; TT = [TT, toc(t0)];
  
  %% Assemble the long vector of weights into structures and evaluate.
  idx = 0;
  D = size(X1,2);
  for j = 1:length(F1)
    if strcmp(F1{j}.type,'conv')
      convdin=F1{j}.filternumrows*F1{j}.filternumcols*F1{j}.numinputmaps;
      convdout=F1{j}.numoutputmaps;
      W_seg=VV(idx+1:idx+(convdin+1)*convdout);
      F1{j}.W=reshape(W_seg,convdin+1,convdout);
      idx=idx+(convdin+1)*convdout;
      D=F1{j}.units;
    else
      units = F1{j}.units;
      W_seg = VV(idx+1:idx+(D+1)*units);
      F1{j}.W = reshape(W_seg,D+1,units);
      idx = idx+(D+1)*units; D = units;
    end
  end
  
  D = size(X2,2);
  for j = 1:length(F2)
    if strcmp(F2{j}.type,'conv')
      convdin=F2{j}.filternumrows*F2{j}.filternumcols*F2{j}.F2{j}.numinputmaps;
      convdout=F2{j}.numoutputmaps;
      W_seg=VV(idx+1:idx+(convdin+1)*convdout);
      F2{j}.W=reshape(W_seg,convdin+1,convdout);
      idx=idx+(convdin+1)*convdout;
      D=F2{j}.units;
    else
      units = F2{j}.units;
      W_seg = VV(idx+1:idx+(D+1)*units);
      F2{j}.W = reshape(W_seg,D+1,units);
      idx = idx+(D+1)*units; D = units;
    end
  end
  
  %% Extract the last CCA step and fix the sign.
  FX1=deepnetfwd_big(X1,F1);
  FX2=deepnetfwd_big(X2,F2);
  [A,B,m1,m2,D]=linCCA(FX1,FX2,K,rcov);
  SIGN=sign(A(1,:)+eps);
  A=A*diag(sparse(SIGN)); B=B*diag(sparse(SIGN));
  A=full(A); B=full(B);
  f1.type='linear'; f1.units=K; f1.W=[A;-m1*A]; F1tmp=F1; F1tmp{end+1}=f1;
  f2.type='linear'; f2.units=K; f2.W=[B;-m2*B]; F2tmp=F2; F2tmp{end+1}=f2;
  
  X_train=deepnetfwd_big(X1,F1tmp);
  X_tune=deepnetfwd_big(XV1,F1tmp);
  CORR1 = [CORR1, DCCA_corr(X_train,deepnetfwd_big(X2,F2tmp),K)]
  CORR2 = [CORR2, DCCA_corr(X_tune,deepnetfwd_big(XV2,F2tmp),K)]
  
  %% I am using the correlation on validation set as criterion for selecting the best networks so far.
  %% You might use other criterion.
  %% In the end, you use F1opt and F2opt as your learned neural networks and test them.
  if CORR2(end)>optvalid
    optvalid=CORR2(end);
    fprintf('Getting better networks!\n');
    F1opt = F1tmp;
    F2opt = F2tmp;
  end
  save(filename,'randseed', 'F1opt','F2opt','optvalid','F1','F2','delta','TT','CORR1','CORR2','mean1','s1','mean2','s2');
end

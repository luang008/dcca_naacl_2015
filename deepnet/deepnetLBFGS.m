function [F,TT,EE] = deepnetLBFGS(X,Y,F,paststeps,batchsize,batchmaxit,tol,maxepoch,saveintr)
% paststeps is the number of past steps to be saved.

% ---------- Argument defaults ----------
if ~exist('batchmaxit','var') || isempty(batchmaxit)
    batchmaxit = 3;
end
if ~exist('tol','var') || isempty(tol) 
    tol = 1e-5; 
end;
if ~exist('maxepoch','var') || isempty(maxepoch) 
    maxepoch = 100; 
end;
if ~exist('saveintr','var') || isempty(saveintr)
    saveintr = 1;
end
% ---------- End of "argument defaults" ----------

Nlayers = length(F);
Ntrain = size(X,1); % Number of training samples.
numbatches = ceil(Ntrain/batchsize);

fprintf('Number of layers: %d\n',Nlayers);
fprintf('Number of training items: %d\n',Ntrain); 
fprintf('Batchsize: %d\n',batchsize); 
fprintf('Number of batches: %d\n',numbatches);

% Initial function and error.
% FF = {F}; 
TT = 0;
[~,EE] = deepnetfwd(X,F,Y);
for k=1:Nlayers
    EE = EE + (F{k}.l)*sum(sum((F{k}.W(1:end-1,:)).^2));
end

VV = []; net = cell(1,Nlayers);
for k=1:Nlayers
    VV = [VV; F{k}.W(:)];
    net{k} = rmfield(F{k},'W'); % net contains information other than weights.
end
fprintf('Number of weight parameters: %d\n',length(VV));

its = 0; cont = (maxepoch>=1);
while cont
    
    for j = 1:numbatches
        % rp = randperm(N);   % Shuffle the data set.
        idx1 = (j-1)*batchsize+1;
        idx2 = min(j*batchsize,Ntrain);
        idx = idx1:idx2;
        Xbatch = X(idx,:);
        Ybatch = Y(idx,:);
        
        [VVVV,~,TIME] =  Olbfgs(@deepnetgrad,...
            {Xbatch,Ybatch,net},VV,tol,batchmaxit,paststeps,saveintr);
        
        % Feed forward all the training data to compute error.
        for m=2:size(VVVV,2)
            VV = VVVV(:,m);
            [~,e,F] = deepnetfwd(X,net,Y,VV);
            for k=1:Nlayers
                e = e + (F{k}.l)*sum(sum((F{k}.W(1:end-1,:)).^2));
            end
            EE = [EE, e];
            % FF = [FF,{F}]; 
        end
        TT = [TT, TIME(2:end)];   % Record the time spent for each epoch.
    end
    
    its = its+1; 
    cont = (its<maxepoch) && (abs(EE(end-1)-EE(end))>=tol*abs(EE(end-1)));
end

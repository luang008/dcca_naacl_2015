function [E,grad] = DCCA_convgrad(VV,X1,X2,F1,F2,dim,rcov,dropprob1,dropprob2)
% VV contains the weight parameters for both networks (view 1 and view 2).
% dim is the number of CCA dimensions to optimize over.
% rcov is the regularization of autocovariance for computing the correlation.

% rng(0);
% STREAM= RandStream.create('mrg32k3a','NumStreams',3,'StreamIndices',2);
% RandStream.setDefaultStream(STREAM)

if ~exist('dropprob1','var') || isempty(dropprob1)
  dropprob1=[0 0*ones(1,length(F1))];
end
if ~exist('dropprob2','var') || isempty(dropprob2)
  dropprob2=[0 0*ones(1,length(F2))];
end

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

[XX1,R1] = forwardpass(X1,F1,dropprob1);
[XX2,R2] = forwardpass(X2,F2,dropprob2);
% Compute objective function and derivative w.r.t. last layer output.
if nargout==1
  corr = DCCA_corr(XX1{end},XX2{end},dim,rcov);
  %%% But we want to minimize the negative correlation.
  E = -corr+R1+R2;
else
  [corr,G1,G2] = DCCA_corr(XX1{end},XX2{end},dim,rcov);
  %%% But we want to minimize the negative correlation.
  E = -corr+R1+R2; G1 = -G1; G2 = -G2;
  
  grad1 = backwardpass(G1,XX1,F1,dropprob1);
  grad2 = backwardpass(G2,XX2,F2,dropprob2);
  grad = [grad1(:); grad2(:)];
end

function [XX,R] = forwardpass(X,F,dropprob)
% Compute all intermediate output and regularization.

N = size(X,1);
Nlayers = length(F);
R = 0;
T = X;
XX = cell(1,Nlayers+1);

% Drop out the inputs.
% j=1; tmp = rand(size(T)); T(tmp<dropprob(j)) = 0; T = T/(1-dropprob(j)); XX{j} = T;
XX{1}=T;
  
% ****** FEED FORWARD ******
for j=1:Nlayers
  
  if strcmp(F{j}.type,'conv')
    % Reshape weights.
    R=R+F{j}.l*sum(sum(F{j}.W(1:end-1,:).^2));
    % Reshape input.
    T=reshape(T,N,F{j}.inputnumrows,F{j}.inputnumcols,F{j}.numinputmaps);
    OUT.respfull_prepool=zeros(N,F{j}.sizeout_prepool1,F{j}.sizeout_prepool2,F{j}.numoutputmaps);
    OUT.respidx=zeros(N,F{j}.sizeout_prepool1,F{j}.sizeout_prepool2,F{j}.numoutputmaps);
    OUT.respfull=zeros(N,F{j}.sizeout1,F{j}.sizeout2,F{j}.numoutputmaps);
    for filteridx=1:F{j}.numoutputmaps
      % Reshape filters, there is one filter for every output feature map.
      Wconv=reshape(F{j}.W(1:end-1,filteridx),1,F{j}.filternumrows,F{j}.filternumcols,F{j}.numinputmaps);
      bconv=F{j}.W(end,filteridx);
      Wconv=Wconv(1,end:-1:1,end:-1:1,end:-1:1);
      % Compute filter response.
      resp=convn(T,Wconv,'valid');
      % Use strides.
      resp=resp(:,1:F{j}.rowstride:end,:);
      resp=resp(:,:,1:F{j}.colstride:end);
      resp=resp+bconv;
      switch F{j}.sigmoid
        case 'sigmoid',
          resp=1./(1+exp(-resp));
        case 'tanh',
          expa=exp(resp); expb=exp(-resp);
          resp=(expa - expb) ./ (expa + expb);
        case 'relu',
          resp(resp<0)=0;
      end
      OUT.respfull_prepool(:,:,:,filteridx)=resp;
      % Start pooling.
      % resp is of dimension [N, sizeout_prepool1, sizeout_prepool2, 1].
      switch F{j}.pooling
        case 'max',
          [resp, respidx]=maxpool(resp, [F{j}.rowpoolratio F{j}.colpoolratio]);
        case 'average',
          [resp, respidx]=avgpool(resp, [F{j}.rowpoolratio F{j}.colpoolratio]);
      end
      OUT.respidx(:,:,:,filteridx)=respidx;
      OUT.respfull(:,:,:,filteridx)=resp;
    end
    OUT.respfull=reshape(OUT.respfull,N,F{j}.sizeout1*F{j}.sizeout2*F{j}.numoutputmaps);
    T=OUT.respfull; XX{j+1}=OUT;
  else
    R = R+F{j}.l*sum(sum(F{j}.W(1:end-1,:).^2)); % Regularization for the weights only.
    T = [T ones(N,1)]*F{j}.W;
    switch lower(F{j}.type)
      case 'linear',
        % Do nothing.
      case 'relu',
        T(T<0) = 0;
      case 'cubic',
        T = nthroot(1.5*T+sqrt(2.25*T.^2+1),3)+nthroot(1.5*T-sqrt(2.25*T.^2+1),3);
        T = real(T);
      case 'sigmoid',
        T = 1./(1+exp(-T));
      case 'tanh',
        expa = exp(T); expb = exp(-T);
        T = (expa - expb) ./ (expa + expb);
      case 'logistic',
        if size(F{j}.W,2)~=1
          error('logistic is only used for binary classification\n');
        else
          T = 1./(1+exp(-T));
        end
      case 'softmax',
        T = exp(T); s = sum(T,2); T = diag(sparse(1./s))*T;
      otherwise,
        error('Invalid layer type: %s\n',F{j}.type);
    end
    % Drop out non-convolutional layer.
    tmp = rand(size(T)); T(tmp<dropprob(j+1)) = 0; T = T/(1-dropprob(j+1)); XX{j+1} = T;
  end
end
% ****** END OF FEED-FORWARD ******

function [dE,delta] = backwardpass(delta,XX,F,dropprob)

% ****** ERROR BACK PROPAGATION ******
Nlayers = length(F);
if Nlayers==0
  dE = [];
else
  % XX is of length Nlayers+1.
  % Last layer, must be non-convolutional.
  j = Nlayers; N=size(delta,1);
  % To take into account the scale issue.
  delta = delta/(1-dropprob(j+1)); T = XX{j+1}*(1-dropprob(j+1)); delta(T==0)=0;
  switch lower(F{j}.type)
    case 'linear',
      % Do nothing.
    case 'relu',
      delta(T<=0) = 0;
    case 'cubic',
      delta = delta./(1+T.^2);
    case 'sigmoid',
      delta = delta.*T.*(1-T);
    case 'tanh',
      delta = delta.*(1-T.^2);
    case 'logistic',
      delta = delta.*T.*(1-T);
    case 'softmax',
      delta = delta.*T - repmat(sum(delta.*T,2),1,size(T,2)).*T;
    otherwise,
      error('Invalid layer type: %s\n',F{j}.type);
  end
  if Nlayers>1 && strcmpi(F{Nlayers-1}.type,'conv');
    de=[XX{j}.respfull ones(N,1)]'*delta;
  else
    de = [XX{j} ones(N,1)]'*delta;
  end
  de(1:end-1,:) = de(1:end-1,:) + 2*F{j}.l*F{j}.W(1:end-1,:);
  dE = de(:);
  % Prepare the delta for next layer.
  delta=delta * F{j}.W(1:end-1,:)';
  
  % Other layers.
  for j = Nlayers-1:-1:1
    T = XX{j+1};
    if strcmp(F{j}.type,'conv')
      % Reshape the deltas to the size of output feature maps. We start from here.
      delta=reshape(delta,N,F{j}.sizeout1,F{j}.sizeout2,F{j}.numoutputmaps);
      % Stretch. Inverse the pooling process.
      delta=delta(:,repmat(1:F{j}.sizeout1,F{j}.rowpoolratio,1),...
        repmat(1:F{j}.sizeout2,F{j}.colpoolratio,1),:);
      delta=delta(:,1:F{j}.sizeout_prepool1,1:F{j}.sizeout_prepool2,:);
      delta=delta.*T.respidx;
      switch F{j}.pooling
        case 'average',
          delta=delta./(F{j}.rowpoolratio*F{j}.colpoolratio);
      end
      % Inverse the nonlinearity.
      switch F{j}.sigmoid
        case 'sigmoid',
          delta=delta.*(T.respfull_prepool).*(1-T.respfull_prepool);
        case 'tanh',
          delta=delta.*(1-(T.respfull_prepool).^2);
        case 'relu',
          delta(T.respfull_prepool==0)=0; % j, length(find(T.respfull_prepool==0))
      end
      % delta is now of size [N, sizeout_prepool1, sizeout_prepool2, numoutputmaps].
      dbias=sum(reshape(delta,N*F{j}.sizeout_prepool1*F{j}.sizeout_prepool2,F{j}.numoutputmaps),1);
      % Fetch the layer below and reshape it to the size of input feature maps.
      if (j>1) && strcmp(F{j-1}.type,'conv')  % There are still conv layers below.
        lowerlayeroutput=XX{j}.respfull;
      else % It is the first layer, lower layer output is the input.
        lowerlayeroutput=XX{j};
      end
      lowerlayeroutput=reshape(lowerlayeroutput,N,F{j}.inputnumrows,F{j}.inputnumcols,F{j}.numinputmaps);
      lowerlayeroutput=repmat(lowerlayeroutput,[1 1 1 1 F{j}.numoutputmaps]);
      % lowerlayeroutput is of size [N inputnumrows intputnumcols numinputmaps numoutputmaps]
      rcW=F{j}.W(1:end-1,:);
      rcW=reshape(rcW,[1, F{j}.filternumrows, F{j}.filternumcols, F{j}.numinputmaps, F{j}.numoutputmaps]);
      rfilter=repmat(rcW,[N, 1, 1, 1, 1]);
      % rfilter is of size [N filternumrow filternumcols numinputmaps numoutputmaps].
      de=zeros(1, F{j}.filternumrows*F{j}.filternumcols*F{j}.numinputmaps,...
        F{j}.numoutputmaps, F{j}.sizeout_prepool1*F{j}.sizeout_prepool2);
      % de is of size [1 filternumrow*filternumcols numinputmaps prepoolimagesize].
      % if (j>1) && strcmp(F{j-1}.type,'conv')  % There are still conv layers below.
      delta_lower=zeros(N,F{j}.inputnumrows,F{j}.inputnumcols,F{j}.numinputmaps);
      % end
      for ai=1:F{j}.sizeout_prepool1
        for aj=1:F{j}.sizeout_prepool2
          % for each pixel in the convolved image.
          acts=reshape(delta(:,ai,aj,:),[N,1,F{j}.numoutputmaps]);
          % find subimage that contribute to the convolution.
          rowstart=(ai-1)*F{j}.rowstride+1;
          rowend=(ai-1)*F{j}.rowstride+F{j}.filternumrows;
          colstart=(aj-1)*F{j}.colstride+1;
          colend=(aj-1)*F{j}.colstride+F{j}.filternumcols;
          inblock=lowerlayeroutput(:, rowstart:rowend, colstart:colend, :, :);
          inblock=bsxfun(@times,reshape(inblock,[N,...
            F{j}.filternumrows*F{j}.filternumcols*F{j}.numinputmaps, F{j}.numoutputmaps]),acts);
          de(:,:,:, (ai-1)*F{j}.sizeout_prepool2+aj)=sum(inblock,1);
          
          % if (j>1) && strcmp(F{j-1}.type,'conv')  % There are still conv layers below.
          delta_lower(:,rowstart:rowend,colstart:colend,:) = ...
            delta_lower(:,rowstart:rowend,colstart:colend,:) + ...
            sum(bsxfun(@times,rfilter,reshape(acts,N,1,1,1,F{j}.numoutputmaps)), 5);
          % end
        end
      end
      de=reshape(sum(de,4),F{j}.filternumrows*F{j}.filternumcols*F{j}.numinputmaps,F{j}.numoutputmaps);
      de=[de; dbias];
      de(1:end-1,:)=de(1:end-1,:) + 2*F{j}.l*F{j}.W(1:end-1,:);
      dE=[de(:); dE];
      % Prepare the delta for next layer.
      delta=delta_lower;
      if j==1 delta=reshape(delta,N,F{j}.inputnumrows*F{j}.inputnumcols*F{j}.numinputmaps); end
    else
      % Non-convolutional layers consider drop out.
      delta = delta/(1-dropprob(j+1)); T = T*(1-dropprob(j+1)); delta(T==0)=0;
      switch lower(F{j}.type)
        case 'linear',
          % Do nothing.
        case 'relu',
          delta(T<=0)=0;
        case 'cubic',
          delta = delta./(1+T.^2);
        case 'sigmoid',
          delta = delta.*T.*(1-T);
        case 'tanh',
          delta = delta.*(1-T.^2);
        case 'logistic',
          delta = delta.*T.*(1-T);
          % Assume that softmax are not used for lower layers.
          % case 'softmax',
          %   delta = delta.*T - repmat(sum(delta.*T,2),1,size(T,2)).*T;
        otherwise,
          error('Invalid layer type: %s\n',F{j}.type);
      end
      if j>1 && strcmpi(F{j-1}.type,'conv');
        de=[XX{j}.respfull ones(N,1)]'*delta;
      else
        de=[XX{j} ones(N,1)]'*delta;
      end
      de(1:end-1,:) = de(1:end-1,:) + 2*F{j}.l*F{j}.W(1:end-1,:);
      dE = [de(:); dE];
      % Prepare the delta for next layer.
      delta=delta * F{j}.W(1:end-1,:)';
    end
  end
  
  % ****** END OF ERROR BACKPROPAGATION ******
end

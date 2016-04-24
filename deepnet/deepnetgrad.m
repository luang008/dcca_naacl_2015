% [E,dE]=deepnetgrad(VV,X,Y,net) computes the training objective function and its gradient for deep neural networks.
%
% In:
%   X: input data, NxD matrix.
%   Y: output data, Nxd matrix.
%   net: a cell array that contains the network architectures. Each element in net
%        specifies the parameters needed for one layer of the deep network.
%        Specifically, it specifies the following
%        type: 'linear', 'sigmoid', 'tanh', 'logistic', 'softmax'
%        units: output dimension of the current layer
%        l: quadratic regularization of the weights for each layer
%   VV: a long vector that contains weights and biases at all layers.
%
% Out:
%   E: training objective function value.
%   dE: derivative of E with respect to the weights VV.

% Copyright (c) 2012 by Miguel A. Carreira-Perpinan and Weiran Wang.

function [E,dE]=deepnetgrad(VV,X,Y,net,dropprob)

if ~exist('dropprob','var') || isempty(dropprob)
  % Last layer (non-hidden) do not drop out.
  dropprob=[0, 0*ones(1,length(net))];
end

% % rng(0);
% STREAM= RandStream.create('mrg32k3a','NumStreams',3,'StreamIndices',2);
% RandStream.setDefaultStream(STREAM)

[N,D]=size(X);
Nlayers=length(net);

if Nlayers==0
  if isempty(VV)
    E=0;
    dE=[];
  else
    error('network and weights should be empty!');
  end
else
  
  % Track dimensions and number of weight parameters that are processed.
  idx=0;
  d0=D;
  % Objective function and itermediate output.
  E=0;
  T=X;
  XX=cell(1,Nlayers); WW=cell(1,Nlayers);
  
  %% ****** FEED FORWARD ******
  % Drop out the inputs.
  % j=1; tmp = rand(size(T)); T(tmp<dropprob(j)) = 0; T = T/(1-dropprob(j)); XX{j} = T;
  XX{1}=T;
  
  for j=1:Nlayers-1
    layer=net{j};
    l=layer.l;
    
    if strcmp(layer.type,'conv')
      % Reshape weights.
      d0=layer.filternumrows*layer.filternumcols*layer.numinputmaps;
      d1=layer.numoutputmaps;
      W=reshape(VV(idx+1:idx+(d0+1)*d1),d0+1,d1);
      E=E+l*sum(sum(W(1:d0,:).^2));
      % Reshape input.
      T=reshape(T,N,layer.inputnumrows,layer.inputnumcols,layer.numinputmaps);
      OUT.respfull_prepool=zeros(N,layer.sizeout_prepool1,layer.sizeout_prepool2,layer.numoutputmaps);
      OUT.respidx=zeros(N,layer.sizeout_prepool1,layer.sizeout_prepool2,layer.numoutputmaps);
      OUT.respfull=zeros(N,layer.sizeout1,layer.sizeout2,layer.numoutputmaps);
      for filteridx=1:layer.numoutputmaps
        % Reshape filters, there is one filter for every output feature map.
        Wconv=reshape(W(1:end-1,filteridx),1,layer.filternumrows,layer.filternumcols,layer.numinputmaps);
        bconv=W(end,filteridx);
        Wconv=Wconv(1,end:-1:1,end:-1:1,end:-1:1);
        % Compute filter response.
        resp=convn(T,Wconv,'valid');
        % Use strides.
        resp=resp(:,1:layer.rowstride:end,:);
        resp=resp(:,:,1:layer.colstride:end);
        resp=resp+bconv;
        switch layer.sigmoid
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
        switch layer.pooling
          case 'max',
            [resp, respidx]=maxpool(resp, [layer.rowpoolratio layer.colpoolratio]);
          case 'average',
            [resp, respidx]=avgpool(resp, [layer.rowpoolratio layer.colpoolratio]);
        end
        OUT.respidx(:,:,:,filteridx)=respidx;
        OUT.respfull(:,:,:,filteridx)=resp;
      end
      OUT.respfull=reshape(OUT.respfull,N,layer.sizeout1*layer.sizeout2*layer.numoutputmaps);
      T=OUT.respfull; XX{j+1}=OUT; WW{j}=W;
      idx=idx+(d0+1)*d1; d0=layer.units;
    else
      % All other types of layers.
      d1=layer.units;
      W=reshape(VV(idx+1:idx+(d0+1)*d1),d0+1,d1);
      E=E+l*sum(sum(W(1:d0,:).^2)); % Regularization for the weights only.
      T=[T ones(N,1)]*W;
      switch lower(layer.type)
        case 'linear',
          % Do nothing.
        case 'relu',
          T(T<0)=0;
        case 'cubic',
          T=nthroot(1.5*T+sqrt(2.25*T.^2+1),3)+nthroot(1.5*T-sqrt(2.25*T.^2+1),3);
        case 'sigmoid',
          T=1./(1+exp(-T));
        case 'tanh',
          expa=exp(T); expb=exp(-T);
          T=(expa - expb) ./ (expa + expb);
        otherwise,
          error('Invalid layer type: %s\n',layer.type);
      end
      % Drop out non-convolutional layer.
      tmp = rand(size(T)); T(tmp<dropprob(j+1)) = 0; T = T/(1-dropprob(j+1));
      XX{j+1}=T; WW{j}=W;
      idx=idx+(d0+1)*d1; d0=d1;
    end
  end
  
  %% Last layer. Compute error and gradient.
  j=Nlayers;
  layer=net{j};
  d1=layer.units;
  l=layer.l;
  W=reshape(VV(idx+1:idx+(d0+1)*d1),d0+1,d1);
  E=E+l*sum(sum(W(1:d0,:).^2)); % Regularization for the weights only.
  T=[T ones(N,1)]*W;
  switch lower(layer.type)
    case 'linear',
      E=E + sum(sum((Y-T).^2));
    case 'relu',
      T(T<0)=0;
      E=E + sum(sum((Y-T).^2));
    case 'cubic',
      T=nthroot(1.5*T+sqrt(2.25*T.^2+1),3)+nthroot(1.5*T-sqrt(2.25*T.^2+1),3);
      E=E + sum(sum((Y-T).^2));
    case 'sigmoid',
      T=1./(1+exp(-T));
      E=E + sum(sum((Y-T).^2));
    case 'tanh',
      expa=exp(T); expb=exp(-T);
      T=(expa - expb) ./ (expa + expb);
      E=E + sum(sum((Y-T).^2));
    case 'logistic',
      if d1~=1
        error('logistic is only used for binary classification\n');
      else
        T=1./(1+exp(-T));
        E=E - sum(Y.*log(T) + (1-Y).*log(1-T));
      end
    case 'softmax',
      T=exp(T); s=sum(T,2); T=diag(sparse(1./s))*T;
      E=E - sum(sum(Y.*log(T))); if isnan(E) error('is nan'); end;
    otherwise,
      error('Invalid layer type: %s\n',layer.type);
  end
  WW{j}=W;
  % ****** END OF FEED-FORWARD ******
  
  %% ****** ERROR BACK PROPAGATION ******
  % Last layer.
  switch lower(layer.type)
    case 'linear',
      delta=2*(T-Y);
    case 'relu',  % Unlikely to happen.
      delta=zeros(size(T));
      delta(T>0)=2*(T(T>0)-Y(T>0));
    case 'cubic',  % Unlikely to happen.
      delta=2*(T-Y)./(1+T.^2);
    case 'sigmoid',
      delta=2*(T-Y).*T.*(1-T);
    case 'tanh',
      delta=2*(T-Y).*(1-T.^2);
    case 'logistic',
      delta=T-Y;
    case 'softmax',
      delta=T-Y;
    otherwise,
      error('Invalid layer type: %s\n',layer.type);
  end
  if Nlayers>1 && strcmpi(net{Nlayers-1}.type,'conv');
    de=[XX{j}.respfull ones(N,1)]'*delta;
  else
    de=[XX{j} ones(N,1)]'*delta;
  end
  de(1:end-1,:)=de(1:end-1,:) + 2*l*W(1:end-1,:);
  dE=de(:);
  % Prepare the delta for next layer.
  delta=delta * W(1:end-1,:)';
  
  %% Other layers.
  for j=Nlayers-1:-1:1
    layer=net{j};
    l=layer.l;
    T=XX{j+1};
    
    if strcmp(layer.type,'conv')
      % Reshape the deltas to the size of output feature maps. We start from here.
      delta=reshape(delta,N,layer.sizeout1,layer.sizeout2,layer.numoutputmaps);
      % Stretch. Inverse the pooling process.
      delta=delta(:,repmat(1:layer.sizeout1,layer.rowpoolratio,1),...
        repmat(1:layer.sizeout2,layer.colpoolratio,1),:);
      delta=delta(:,1:layer.sizeout_prepool1,1:layer.sizeout_prepool2,:);
      delta=delta.*T.respidx;
      switch layer.pooling
        case 'average',
          delta=delta./(layer.rowpoolratio*layer.colpoolratio);
      end
      % Inverse the nonlinearity.
      switch layer.sigmoid
        case 'sigmoid',
          delta=delta.*(T.respfull_prepool).*(1-T.respfull_prepool);
        case 'tanh',
          delta=delta.*(1-(T.respfull_prepool).^2);
        case 'relu',
          delta(T.respfull_prepool==0)=0;
      end
      % delta is now of size [N, sizeout_prepool1, sizeout_prepool2, numoutputmaps].
      dbias=sum(reshape(delta,N*layer.sizeout_prepool1*layer.sizeout_prepool2,layer.numoutputmaps),1);
      % Fetch the layer below and reshape it to the size of input feature maps.
      if (j>1) && strcmp(net{j-1}.type,'conv')  % There are still conv layers below.
        lowerlayeroutput=XX{j}.respfull;
      else % It is the first layer, lower layer output is the input.
        lowerlayeroutput=XX{j};
      end
      lowerlayeroutput=reshape(lowerlayeroutput,N,layer.inputnumrows,layer.inputnumcols,layer.numinputmaps);
      lowerlayeroutput=repmat(lowerlayeroutput,[1 1 1 1 layer.numoutputmaps]);
      % lowerlayeroutput is of size [N inputnumrows intputnumcols numinputmaps numoutputmaps]
      rcW=WW{j}(1:end-1,:);
      rcW=reshape(rcW,[1, layer.filternumrows, layer.filternumcols, layer.numinputmaps, layer.numoutputmaps]);
      rfilter=repmat(rcW,[N, 1, 1, 1, 1]);
      % rfilter is of size [N filternumrow filternumcols numinputmaps numoutputmaps].
      de=zeros(1, layer.filternumrows*layer.filternumcols*layer.numinputmaps,...
        layer.numoutputmaps, layer.sizeout_prepool1*layer.sizeout_prepool2);
      % de is of size [1 filternumrow*filternumcols numinputmaps prepoolimagesize].
      if (j>1) && strcmp(net{j-1}.type,'conv')  % There are still conv layers below.
        delta_lower=zeros(N,layer.inputnumrows,layer.inputnumcols,layer.numinputmaps);
      end
      for ai=1:layer.sizeout_prepool1
        for aj=1:layer.sizeout_prepool2
          % for each pixel in the convolved image.
          acts=reshape(delta(:,ai,aj,:),[N,1,layer.numoutputmaps]);
          % find subimage that contribute to the convolution.
          rowstart=(ai-1)*layer.rowstride+1;
          rowend=(ai-1)*layer.rowstride+layer.filternumrows;
          colstart=(aj-1)*layer.colstride+1;
          colend=(aj-1)*layer.colstride+layer.filternumcols;
          inblock=lowerlayeroutput(:, rowstart:rowend, colstart:colend, :, :);
          inblock=bsxfun(@times,reshape(inblock,[N,...
            layer.filternumrows*layer.filternumcols*layer.numinputmaps, layer.numoutputmaps]),acts);
          de(:,:,:, (ai-1)*layer.sizeout_prepool2+aj)=sum(inblock,1);
          
          if (j>1) && strcmp(net{j-1}.type,'conv')  % There are still conv layers below.
            delta_lower(:,rowstart:rowend,colstart:colend,:) = ...
              delta_lower(:,rowstart:rowend,colstart:colend,:) + ...
              sum(bsxfun(@times,rfilter,reshape(acts,N,1,1,1,layer.numoutputmaps)), 5);
          end
        end
      end
      de=reshape(sum(de,4),layer.filternumrows*layer.filternumcols*layer.numinputmaps,layer.numoutputmaps);
      de=[de; dbias];
      de(1:end-1,:)=de(1:end-1,:) + 2*l*WW{j}(1:end-1,:);
      dE=[de(:); dE];
      if (j>1) && strcmp(net{j-1}.type,'conv')  % There are still conv layers below.
        delta=delta_lower;
      end
    else
      % Non-convolutional layers consider drop out.
      delta = delta/(1-dropprob(j+1)); T = T*(1-dropprob(j+1)); delta(T==0)=0;
      switch lower(layer.type)
        case 'linear',
          % Do nothing.
        case 'relu',
          delta(T<=0)=0;
        case 'cubic',
          delta=delta./(1+T.^2);
        case 'sigmoid',
          delta=delta.*T.*(1-T);
        case 'tanh',
          delta=delta.*(1-T.^2);
        otherwise,
          error('Invalid layer type: %s\n',layer.type);
      end
      if j>1 && strcmpi(net{j-1}.type,'conv');
        de=[XX{j}.respfull ones(N,1)]'*delta;
      else
        de=[XX{j} ones(N,1)]'*delta;
      end
      de(1:end-1,:)=de(1:end-1,:) + 2*l*WW{j}(1:end-1,:);
      dE=[de(:); dE];
      % Prepare the delta for next layer.
      if j>1
        delta=delta * WW{j}(1:end-1,:)';
      end
    end
  end
  % ****** END OF ERROR BACKPROPAGATION ******
  
end

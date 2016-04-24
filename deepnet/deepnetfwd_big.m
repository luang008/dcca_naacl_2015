function [T,e,F] = deepnetfwd_big(X,F,Y,W)

if length(F)==0
  T=X;
else

if nargin>3
  D = size(X,2); idx = 0;
  for j = 1:length(F);
    units = F{j}.units;
    W_seg = W(idx+1:idx+(D+1)*units);
    F{j}.W = reshape(W_seg,D+1,units);
    idx = idx+(D+1)*units; D = units;
  end
end

N=size(X,1);
D=F{end}.units;
T=zeros(N,D);

B=5000; % Block size.
NUMBATCHES=ceil(N/B);

for i=1:NUMBATCHES
  startidx=(i-1)*B+1;
  endidx=min(i*B,N);
  
  Tseg = deepnetfwd(X(startidx:endidx,:),F);
  T(startidx:endidx,:)=Tseg;
end
  
if (nargin>2) && (nargout>1)
  switch F{end}.type
    case 'linear',
      e = sum(sum((Y-T).^2));
    case 'relu',  % Unlikely to happen.
      e = sum(sum((Y-T).^2));
    case 'cubic',  % Unlikely to happen.
      e = sum(sum((Y-T).^2));
    case 'sigmoid',
      e = sum(sum((Y-T).^2));
    case 'tanh',
      e = sum(sum((Y-T).^2));
    case 'logistic',
      e = sum(-Y.*log(T)-(1-Y).*log(1-T));
    case 'softmax',
      T(T==0)=eps;
      e = sum(sum(-Y.*log(T)));
    otherwise,
      error('Invalid layer type: %s\n',F{end}.type);
  end

end
end

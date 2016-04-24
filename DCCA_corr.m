function [corr,grad1,grad2]=DCCA_corr(H1,H2,dim,r)
% Compute the delta at the last layer.
% dim is the desired dimensionality of CCA space.
% r is the regularization of autocovariance for computing the correlation.

if ~exist('r','var') || isempty(r)
  r = [0 0];
else
  if numel(r)==1
    r = [r r];
  end
end

[N,d1] =size(H1);
[~,d2] =size(H2);
% Remove mean.
m1 = mean(H1,1); H1 = H1-repmat(m1,N,1);
m2 = mean(H2,1); H2 = H2-repmat(m2,N,1);

S11 = (H1'*H1)/(N-1)+r(1)*eye(d1); S22 = (H2'*H2)/(N-1)+r(2)*eye(d2); 
S12 = (H1'*H2)/(N-1);
[V1,D1] = eig(S11); [V2,D2] = eig(S22);
% For numerical stability.
D1 = diag(D1); idx1 = find(D1>1e-12); D1 = D1(idx1); V1 = V1(:,idx1);
D2 = diag(D2); idx2 = find(D2>1e-12); D2 = D2(idx2); V2 = V2(:,idx2);

K11 = V1*diag(D1.^(-1/2))*V1';
K22 = V2*diag(D2.^(-1/2))*V2';
T = K11*S12*K22;
[U,D,V] = svd(T,0); 
U = U(:,1:dim); D = D(1:dim,1:dim); V = V(:,1:dim);
corr = sum(sum(D));

if nargout>1
  DELTA12 = (K11*U)*(V'*K22);
  DELTA11 = -0.5*(K11*U)*D*(U'*K11);
  DELTA22 = -0.5*(K22*V)*D*(V'*K22);
  
  grad1 = 2*H1*DELTA11+H2*DELTA12'; grad1 = grad1/(N-1);
  grad2 = H1*DELTA12+2*H2*DELTA22; grad2 = grad2/(N-1);
end

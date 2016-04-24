% Linaer CCA used as baseline
function [A,B,m1,m2,D]=linCCA(H1,H2,dim,r)
% H1 and H2 are NxD matrices containing samples rowwise.
% dim is the desired dimensionality of CCA space.
% r is the regularization of autocovariance for computing the correlation.
% A and B are the transformation matrix for view 1 and view 2.
% m1 and m2 are the mean for view 1 and view 2.
% D is the vector of singular values.
disp('lincca')
disp(dim)
  disp(size(H1))
  disp(size(H2))
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
D = diag(D);
disp(size(U));
disp(dim);
A = K11*U(:,1:dim);
B = K22*V(:,1:dim);
D = D(1:dim);

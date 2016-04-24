function X=normr(X)

s=sqrt(sum(X.^2,2)+eps);
X=bsxfun(@times,X,1./s);
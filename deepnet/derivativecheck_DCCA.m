% clear all;
N=2;

Din=4;
X1=rand(N,Din);
DD=[3,3];
j=1; F1{j}.type='sigmoid'; F1{j}.l=0.1; F1{j}.units=DD(j); F1{j}.W=randn(Din+1,DD(j))*0.1;
j=2; F1{j}.type='linear'; F1{j}.l=0.3; F1{j}.units=DD(j); F1{j}.W=randn(DD(j-1)+1,DD(j))*0.1;

Din=3;
X2=rand(N,Din);
DD=[3,3];
j=1; F2{j}.type='tanh'; F2{j}.l=5.0; F2{j}.units=DD(j); F2{j}.W=randn(Din+1,DD(j))*0.1;
j=2; F2{j}.type='linear'; F2{j}.l=2.0; F2{j}.units=DD(j); F2{j}.W=randn(DD(j-1)+1,DD(j))*0.1;

VV=[];
for j=1:length(F1)
  VV = [VV; F1{j}.W(:)];
end
derivativeCheck(@deepnetgrad,VV(:),1,2,X1,[1 0 0; 0 1 0],F1)
derivativeCheck(@deepnetgrad_dropout,VV(:),1,2,X1,[1 0 0; 0 1 0],F1,[0.2 0.5]);
[aa,bb]=deepnetgrad_dropout(VV(:),X1,[1 0 0; 0 1 0],F1,[0.2 0.5]);

for j=1:length(F2)
  VV = [VV; F2{j}.W(:)];
end

K=1;
order = 1;
derivativeCheck(@DCCA_grad,VV(:),order,2,X1,X2,F1,F2,K,[0.7 5])
derivativeCheck(@DCCA_grad,VV(1:27),order,2,X1,X2,F1,{},K,[4.7 0.5])
derivativeCheck(@DCCA_grad,VV(28:end),order,2,X1,X2,{},F1,K,[0.7 3])
derivativeCheck(@DCCA_grad_dropout,VV(:),order,2,X1,X2,F1,F2,K,[0.7 5])
[aa,bb]=DCCA_grad_dropout(VV(:),X1,X2,F1,F2,K,[0.7 5]);

Y1=[1 0 0; 0 1 0];
Y2=[1 0 0; 0 0 1];
Din = 3;
DD=[2,3];
j=1; G1{j}.type='sigmoid'; G1{j}.l=0.2; G1{j}.units=DD(j); G1{j}.W=randn(Din+1,DD(j))*0.1;
j=2; G1{j}.type='softmax'; G1{j}.l=0.3; G1{j}.units=DD(j); G1{j}.W=randn(DD(j-1)+1,DD(j))*0.1;
Z1=randn(10,4);

Din=3;
DD=[2,3];
j=1; G2{j}.type='cubic'; G2{j}.l=5.0; G2{j}.units=DD(j); G2{j}.W=randn(Din+1,DD(j))*0.1;
j=2; G2{j}.type='softmax'; G2{j}.l=2.0; G2{j}.units=DD(j); G2{j}.W=randn(DD(j-1)+1,DD(j))*0.1;
Z2=randn(10,3);

VV=[];
for j=1:length(F1)
  VV = [VV; F1{j}.W(:)];
end
for j=1:length(G1)
  VV = [VV; G1{j}.W(:)];
end
for j=1:length(F2)
  VV = [VV; F2{j}.W(:)];
end
for j=1:length(G2)
  VV = [VV; G2{j}.W(:)];
end

order = 1;
lambda = 1.8;
derivativeCheck(@DCCA_sup,VV(:),order,2,X1,Y1,X2,Y2,lambda,Z1,Z2,F1,G1,F2,G2,[0.7 5])
derivativeCheck(@DCCA_sup,VV(1:68),order,2,X1,Y1,[],[],lambda,Z1,Z2,F1,G1,F2,{},[0.6 2])
derivativeCheck(@DCCA_sup,VV(1:44),order,2,X1,Y1,[],[],lambda,Z1,Z2,F1,G1,{},{},[0.6 2])
% % % derivativeCheck(@DCCA_sup,VV(28:44),order,2,X1,Y1,[],[],lambda,Z1,Z2,{},G1,{},{},[0 3])

% % % H1=randn(5,2);
% % % H2=randn(5,2);
% % % 
% % % order = 1;
% % % derivativeCheck(@DCCAwrapper_corr,[H1(:);H2(:)],order,2,5,2,53)


function [a,r,spectrum] = nystromsolve(X,Y,sample,sigma,gamma)

[n,~] = size(X);
[U,L] = nystromeig(X,sigma,sample);
[Q,R] = qr(U,0);

%Eigendecompose inner matrix R L R' = Qs S Qs', diagonalize
[Qs,S] = eig(R*diag(L)*R');
S = diag(S);
qy = Qs'*(Q'*Y);
aq = qy./S;  
%regularization
k = find(S>gamma,1,'last');
aq(k:end)=0;
a = Q*(Qs*aq);
% residual
r = Q*(Qs*(S.*aq)) - Y;


spectrum = S;


function [alpha] = solveQR(Q,R,L,reg,Y)
%solveQR Function solves the kernel system (K)alpha = Y
%functionality for the Laplacian can be added later. The solution is of
%the form: alpha = Q (R L R^T)^-1 Q^T f. We use the reg parameter to
%determine how many elements of the new diagonal S are kept

%Reduce QR
[~,n] = size(R);
Q = Q(:,1:n);
R = R(1:n,:);

%Form matrix R L R^T and factorize
[Qs,S] = eig(R*diag(L)*R');
Qs = Qs(:,end:-1:end-reg+1);
S = diag(S); S = S(end:-1:end-reg+1);

%Compute alpha = Q Qs S^-1 Qs' Q' Y
rhs = Qs'*Q'*Y;
alpha = Q*(Qs*(rhs./S));


end


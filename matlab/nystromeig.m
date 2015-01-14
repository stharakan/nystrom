function [Un, Ln] = nystromeig(X,sigma, sample,p)
%NYSTROMEIG Computes the nystrom approximation to the kernel gram matrix K.
%The paper we are following allows a further time speed up by introducing p
%<= m but for now we assume p = m (so they are interchangeable for now).
%
%   INPUTS:
%       - X: n x d matrix of n points sitting in d-dim feature space.
%       - m: size of the system to be solved exactly
%       - p: portion of full m decomposition to use
%       - ll: number  of labeled points per class (vector)
%       - pick: decides how to choose the m points for the following values
%           'random' = randomly pick out of the n points in X
%           'kmeans' = kmeans-random from m clusters
%       
%   OUTPUTS:
%       - Un:  n x p matrix consisting of first p columns of the Un used to
%       approximate U in the eigendecomposition of K.
%       - Ln: p-vector list of the eigenvalues corresponding to the p
%       eigenvectors of Un

[n,d] = size(X);
m = length(sample);
if ~exist('p')
				p=m;
end

idx_m = sample;

% %random choice
% if strcmp(pick,'random')
%     idx_m = randpick(m,n);
% elseif n==m
%   idx_m = [1:n]';
% %kmeans
% else strcmp(pick,'kmeans')
%     idx_m = kmeanspick(X,m,ll);
% end

% xnorms = sum(X.*X,2); % n x 1 vector of square norms of X
% K_nm1 = exp(-(repmat(xnorms,1,m) + repmat(xnorms(idx_m)',n,1) ...
%     - 2.*X*X(idx_m,:)')./(2*sigma^2));
K_nm=kernel(X,X(sample,:),sigma);




% [Um,Lm] = eig(K_nm(idx_m,:));
% Ln = (n/m).*diag(Lm);
% Ln = Ln(m:-1:m-p+1);
% 
% Un = (sqrt(n/m)).*((K_nm*Um(:,m:-1:m-p+1))*(diag(1./Ln)));

[Um,Lm,~]=svd(K_nm(idx_m,:));

nm = single(n/m);
Ln = nm.*diag(Lm);
clear Lm

Ln = Ln(1:p);
Um = Um(:,1:p);
Um = Um*diag(sqrt(nm)./Ln);
Un= K_nm*Um;

%Ln = Ln(1:p);
%Un = Un(:,1:p);

end

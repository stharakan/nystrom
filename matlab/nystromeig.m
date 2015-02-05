function [Un, Ln,varargout] = nystromeig(X,sigma, sample,p,flag)
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
if ~exist('flag')
    flag = 1;
end

idx_m = sample;

K_nm=kernel(X,X(sample,:),sigma);

[Um,Lm,~]=svd(K_nm(idx_m,:));

nm = n/m;
Ln = nm.*diag(Lm);
clear Lm

Ln = Ln(1:p);
Um = Um(:,1:p);
if flag
    %Um = Um*diag(sqrt(nm) * ones(size(Ln)));
    Um = Um*diag(sqrt(nm)./Ln);
    Un= K_nm*Um;
else
    Un = 0;
end

varargout{1} = Um;
end

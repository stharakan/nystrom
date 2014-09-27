function [output] = testdata(alpha,sigma,Xtrain,Xtest)
%TESTDATA: Runs the classifier on the test data set Xtest given the alpha
%that define f in terms of the kernel. We assume the kernel is exponential
%based, with sigma as the width parameter. Eventually add functionality to 
%display results in some manner. 
%
%   INPUTS:
%       - alpha: coefficients of K(x,xi) in the expression for f, arranged
%       in a column vector of length n (number of elements in Xtrain)
%       - sigma: parameter for the kernel fucntion K, determining width
%       - Xtrain: training data set represented as a n x d matrix of n
%       points sitting in d dimentional space.
%       - Xtest: test set to be evaluated, represented as a m x d matrix of
%       vectors
%
%   OUTPUTS:
%       - output: vector of length m giving the results of running the
%       classifier on each row vector of Xtest

[n,~] = size(Xtrain);
[m,~] = size(Xtest);

trnorms = sum(Xtrain.*Xtrain,2);
tenorms = sum(Xtest.*Xtest,2);
K = exp(-(repmat(tenorms,1,n) + repmat(trnorms',m,1) ...
    - 2*Xtest*Xtrain')/(2*sigma^2));


output = K*alpha;
end


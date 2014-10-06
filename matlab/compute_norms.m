% script to estimate the norm of the approximate kernel matrix used by the 
% Nystrom method -- ready for use on the following files:
%
% cpusmall - 'cpusmall' NEEDS TO BE SCALED!!
% winequality - 'wine'
% synthetic spiral - 'spiral'
% SUSY - 'susy' 
% covertype - 'covtype'
% ijcnn1 - 'ijcnn1'
% mnist8m - 'mnist8m' --> not yet working

% Load data
file = 'ijcnn1';
dir = '/org/groups/padas/lula_data/machine_learning/';

tic;
[X,~,~,~,~] = loaddata(file,dir);
toc

[N,d] = size(X);
nystrom_rank = 128;
nystrom_m = 2*nystrom_rank;
sample_method = 'random';
sigma = sigma_given(file); 

% create function handle
disp('-----Nystrom decmp-------')
tic;
sample = createsample(X,nystrom_m,[],sample_method);
[U, L] = nystromeig(X, sigma, sample,nystrom_rank);
toc
matvec = @(rhs) NystromMatVec(U, L, rhs);

% Estimate Norms
num_norm_samples = 10;
disp('-----Estimate norm------')
tic;
est_norm = Estimate2Norm(matvec, num_norm_samples, N);
toc

%Compute true norm

% Output results
fprintf('Est norm: %.15f\n', est_norm);
%fprintf('True norm: %.15f\n', true_norm);















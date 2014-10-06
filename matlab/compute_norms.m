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
norm_sample_size = 1000;

% create function handle
disp('-----Nystrom decmp-------')
tic;
sample = createsample(X,nystrom_m,[],sample_method);
[U, L] = nystromeig(X, sigma, sample,nystrom_rank);
toc
matvec = @(rhs) NystromMatVec(U, L, rhs);

% Estimate Norms
disp('-----Estimate norm------')
%num_norm_samples = 10;
%tic;
%est_norm = Estimate2Norm(matvec, num_norm_samples, N);
%toc
smpidx = randperm(N);
smpidx = smpidx(1:norm_sample_size);
tic;
w = ones(N,1)./sqrt(N);
estKw = U(smpidx,:) * ( L.* (U'*w));
truKw = kernel(X(smpidx,:),X,sigma)*w;
rel_error = norm(estKw - truKw)/norm(truKw);
toc

% Output results
fprintf('Rel error: %.15f\n', rel_error);
%fprintf('Est norm: %.15f\n', est_norm);
%fprintf('True norm: %.15f\n', true_norm);















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
%
%%%%%  ------- %%%%%%%%
% ANY .askit FILE - GIVE EXACT FILE NAME
%%%%%  ------- %%%%%%%%

% Load data
file = 'gaussian_16d_100K.askit';
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
w = ones(N,1)./sqrt(N);
uw = L.*(U'*w);

tic; 
%newN = norm_sample_size/10;
%numerator = 0;
%denom = 0;
%for i = 1:10
%				sidx = smpidx(((i-1)*newN+1):i*newN);
%				estKw = U(sidx,:) *uw;
%				truKw = kernel(X(sidx,:),X,sigma)*w;
%				numerator = numerator + norm(estKw - truKw)^2;
%				denom = denom + norm(truKw)^2;				
%end
%rel_error = sqrt(numerator/denom);

estKw = U(smpidx,:) * uw;
truKw = kernel(X(smpidx,:),X,sigma)*w;
rel_error = norm(truKw - estKw) / norm(truKw);

toc

% Output results
fprintf('Rel error: %.15f\n', rel_error);















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
file = 'susy_scaled.askit';
nystrom_rank = 2048;
sigma_choice = 4;
flag = 1; % 0=data_loaded, 1=need to load
dir = '/org/groups/padas/lula_data/machine_learning/';

if flag
    clearvars -except dir sigma_choice nystrom_rank file
    [X,~,~,~,~] = loaddata(file,dir);
else
    clearvars -except X dir sigma_choice nystrom_rank file
    disp(['Using previously loaded data ', file]);
end

[N,d] = size(X);
X = single(X);
nystrom_m = 2*nystrom_rank;
sample_method = 'random';
sigma = sigma_given(file,sigma_choice);
norm_sample_size = 1000;
runs = 1;

% create function handle
disp('-----Nystrom decmp-------')
tic;
sample = createsample(X,nystrom_m,[],sample_method);
[U, L] = nystromeig(X, sigma, sample,nystrom_rank);
toc
matvec = @(rhs) NystromMatVec(U, L, rhs);

% Estimate Norms
disp('-----Estimate norm------')

smpidx = randperm(N);
smpidx = smpidx(1:norm_sample_size);
w = ones(N,1)./sqrt(N);
uw = L.*(U'*w);

if(runs ~=1)
    rel_error = 0;
    newN = norm_sample_size/runs;
    for i = 1:runs
        sidx = smpidx(((i-1)*newN+1):i*newN);
        estKw = U(sidx,:) *uw;
        truKw = kernel(X(sidx,:),X,sigma)*w;
        rel_error = rel_error + sum(abs((estKw - truKw)./truKw))/norm_sample_size;
    end
else
    estKw = U(smpidx,:) * uw;
    truKw = kernel(X(smpidx,:),X,sigma)*w;
    rel_error = sum(abs((truKw - estKw)./truKw))/norm_sample_size;
end

% Output results
fprintf('Sigma: %.4f , Rank: %d\n', sigma,nystrom_rank);
fprintf('Rel error: %.15f\n', rel_error);




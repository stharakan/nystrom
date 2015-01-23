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
file = 'covtype_scaled.askit';
nystrom_rank = 256;
sigma_choice = 3;
flag = 0; % 0=data_loaded, 1=need to load
dir = '/org/groups/padas/lula_data/machine_learning/';
%dir ='/h2/sameer/Documents/research/nystrom/';

if flag
    clearvars -except dir sigma_choice nystrom_rank file sample
    [X,Y,Xtest,Ytest,~] = loaddata(file,dir);
else
    clearvars -except X sigma_choice nystrom_rank file sample
    disp(['Using previously loaded data ', file]);
end

sflag = 1; %0=sample loaded, 1=need to load
eflag = 1; %0=dont compute errors, 1= compute them
[N,d] = size(X);
X = single(X);
nystrom_m = 2*nystrom_rank;
sample_method = 'random';
sigma = sigma_given(file,sigma_choice)
norm_sample_size = 1000;
runs = 1;

disp('-----Nystrom decmp-------')
tic;
if sflag
	sample = createsample(X,nystrom_m,[],sample_method);
else
	disp('Using old sample');
end
[U, L] = nystromeig(X, sigma, sample,nystrom_rank);
toc
matvec = @(rhs) NystromMatVec(U, L, rhs);

% Estimate Norms
disp('-----Estimate norm------')
if eflag
	[abs_error, rel_error] = matvec_errors(X,U,L,sigma,norm_sample_size,1);
	% Output results
	fprintf('Sigma: %.4f , Rank: %d\n', sigma,nystrom_rank);
	fprintf('Rel error: %.15f\n', rel_error);

else
	disp('Not computing errors')
end




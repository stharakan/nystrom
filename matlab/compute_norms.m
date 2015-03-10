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
file = 'covtype_libsvm';
nystrom_rank = 16384;
%sigma_choice = 3;
%sigma = sigma_given(file,sigma_choice)
sigma = 0.16
digit_chosen = 1;
flag = 1; % 0=data_loaded, 1=need to load
dir = '/org/groups/padas/lula_data/machine_learning/';
%dir ='/h2/sameer/Documents/research/nystrom/';
dir = '/work/00921/biros/maverick/data/machine_learning/';

if flag
    clearvars -except dir sigma nystrom_rank file sample digit_chosen
    [X,Y,Xtest,Ytest,~] = loaddata(file,dir,digit_chosen);
else
    clearvars -except X sigma nystrom_rank file sample
    disp(['Using previously loaded data ', file]);
end

kde = 0; %do kernel density estimation
approx = 0; %approximate or exact kernel 
do_all = 1; %all test points or 1000 samples
sflag = 1; %0=sample loaded, 1=need to load
eflag = 1; %0=dont compute errors, 1= compute them
[N,d] = size(X);
nystrom_m = nystrom_rank;
sample_method = 'random';
norm_sample_size = 1000;
runs = 1;

disp('-----Nystrom decmp-------')
tic;
if approx || eflag
    if sflag
        sample = createsample(X,nystrom_m,[],sample_method);
    else
	    disp('Using old sample');
    end 
    [U, L,Um] = nystromeig(X, sigma, sample,nystrom_rank,1);
    matvec = @(rhs) NystromMatVec(U, L, rhs);
else
    disp('No Nystrom approximation formed');
end
toc


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

if kde
    w = gen_kde_weights(Y);
    if approx 
    	[rel_error,class_corr] = regress_errors(X(sample,:),Xtest,Ytest,w,sigma,Um,U,L,do_all);
    else
        [rel_error,class_corr] = regress_errors(X,Xtest,Ytest,w,sigma);
    end
    rel_error
    class_corr
end





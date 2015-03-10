%%%%%  ------- %%%%%%%%
% ANY libsvm FILE - GIVE FILE UP TO _libsvm
%%%%%  ------- %%%%%%%%

%% File load parameters(if necessary)
flag = 1; % 0=data_loaded, 1=need to load
file = 'covtype_libsvm';
dir = '/org/groups/padas/lula_data/machine_learning/';
%dir ='/h2/sameer/Documents/research/nystrom/';
dir = '/work/00921/biros/maverick/data/machine_learning/';

if flag
    clearvars -except dir file sample lambda U L Q D 
    [X,Y,Xtest,Ytest,~] = loaddata(file,dir);
else
    clearvars -except X Y Xtest Ytest file sample lambda U L Q D 
    disp(['Using previously loaded data ', file]);
end

%% Specify other parameters
nystrom_rank = 16384;
sigma_choice = 2;
sigma = sigma_given(file,sigma_choice);
sigma = 0.35
sflag = 1; %0=use old sample  , 1=generate new
lflag = 0; lambda = 0.005; 
nflag = 1; %0=nystrom computed, 1=generate new
eflag = 1;
rflag = 0; 
pflag = 1;
do_all = 0;
norm_sample_size = 1000;
runs = 1;
sample_method = 'random';


%% Select Lambda
if lflag
disp('------Lambda Cross-Val------');
tic;
[lambda,spectrum,err,errors] = cv_lambda(X,Y,sigma,nystrom_rank,pflag);
toc
disp(['Lamba chosen: ',num2str(lambda),'; Associated error: ', num2str(err)]);
else
	if ~exist('lambda')
		lambdastr = 'undefined';
	else
		lambdastr = num2str(lambda);
	disp(['Not running cross-val, previous computed lambda is ', lambdastr]);
end
end

%% Full Nystrom decomposition
disp('-----Nystrom decmp-------')
[N,d] = size(X);
nystrom_m = nystrom_rank;

if nflag
    tic;
    
    %pick sample if needed
    if sflag
        sample = createsample(X,nystrom_m,[],sample_method);
    else
        disp('Using previous sample');
    end
    
    if do_all
        [U, L,Um] = nystromeig(X, sigma, sample,nystrom_rank,1);
    else
        [U, L] = nystromeig(X, sigma, sample,nystrom_rank,1);
    end
    
    toc
	%orthogonalize
    [Q,D] = nystromorth(U,L);
	perc_spectrum_kept = sum(D>lambda)/length(D)
else
    disp('Using previous nystrom decomp');
end

matvec = @(rhs) NystromMatVec(U, L, rhs);



%% Estimate Errors
disp('-----Estimating kernel approx error------')

%Mat-vec error
tic;
if eflag
[ptwise_err_mv,rel_err_mv] = matvec_errors(X,U,L,sigma,norm_sample_size,runs); 
end
toc

% Output results
fprintf('Sigma: %.4f , Rank: %d\n', sigma,nystrom_rank);
fprintf('Rel error: %.15f\n', rel_err_mv);

if (~(isempty(Xtest) || isempty(Ytest)) && rflag) 
    disp('-----Estimating test set errors------');    
   	tic;

    %find weight vector
    [w] = find_weights(Q,D,Y,lambda);
    %compute errors
    
    if do_all
        [Ntest,~] = size(Xtest);
        idx = floor(1: (Ntest/1000) : Ntest);
	idx = 1:Ntest;
        [relErr_approx,class_corr] = regress_errors(X(sample,:),Xtest(idx,:),Ytest,w,sigma,Um,U,L,1);
    else
        [relErr_approx,class_corr] = regress_errors(X,Xtest,Ytest,w,sigma);
    end
    toc
    relErr_approx
    class_corr
    %Absolute, relative with exact computation
    
else
    disp('No test set comparison available.');

end





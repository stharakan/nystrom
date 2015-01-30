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
nystrom_rank = 1024;
sigma_choice = 2;
sflag = 1; %0=use old sample  , 1=generate new
lflag = 1; %0=lambda computed , 1=need to compute
nflag = 1; %0=nystrom computed, 1=generate new
cvflag = 1; 
pflag = 1;
norm_sample_size = 1000;
runs = 1;
sample_method = 'random';
sigma = sigma_given(file,sigma_choice);

%% Select Lambda
if cvflag
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
nystrom_m = 2*nystrom_rank;

if nflag
    tic;
    
    %pick sample if needed
    if sflag
        sample = createsample(X,nystrom_m,[],sample_method);
    else
        disp('Using previous sample');
    end
    
    [U, L] = nystromeig(X, sigma, sample,nystrom_rank,1);
    
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
[ptwise_err_mv,rel_err_mv] = matvec_errors(X,U,L,sigma,norm_sample_size,runs); 
toc

% Output results
fprintf('Sigma: %.4f , Rank: %d\n', sigma,nystrom_rank);
fprintf('Rel error: %.15f\n', rel_err_mv);

if ~(isempty(Xtest) || isempty(Ytest))
    disp('-----Estimating test set errors------');    
   	tic;

    %find weight vector
    w = find_weights(Q,D,Y,lambda);
    
    %compute errors
    [absErr_approx, relErr_approx,class_corr] = regress_errors(X,Xtest,Ytest,w,sigma,norm_sample_size);
	toc
    
    absErr_approx
    relErr_approx
	class_corr
    %Absolute, relative with exact computation
    
else
    disp('No test set comparison available.');

end





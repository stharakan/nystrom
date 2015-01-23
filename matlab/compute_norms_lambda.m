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
    clearvars -except dir file sample lambda U L
    [X,Y,Xtest,Ytest,~] = loaddata(file,dir);
else
    clearvars -except X Y Xtest Ytest file sample lambda U L 
    disp(['Using previously loaded data ', file]);
end

%% Specify other parameters
nystrom_rank = 512;
sigma_choice = 2;
sflag = 1; %0=use old sample  , 1=generate new
lflag = 1; %0=lambda computed , 1=need to compute
nflag = 1; %0=nystrom computed, 1=generate new
cvflag = 0; 
norm_sample_size = 1000;
runs = 1;
sample_method = 'random';
sigma = sigma_given(file,sigma_choice);

%% Select Lambda
if cvflag
disp('------Lambda Cross-Val------');
tic;
[lambda,spectrum,err] = cv_lambda(X,Y,sigma,nystrom_rank);
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
else
    disp('Using previous nystrom decomp');
end

matvec = @(rhs) NystromMatVec(U, L, rhs);



%% Estimate Errors
disp('-----Estimating kernel approx error------')

%Mat-vec error
tic;
[abs_err_mv,rel_err_mv] = matvec_errors(X,U,L,norm_sample_size,runs); 
toc

% Output results
fprintf('Sigma: %.4f , Rank: %d\n', sigma,nystrom_rank);
fprintf('Rel error: %.15f\n', rel_err_mv);

if ~(isempty(Xtest) || isempty(Ytest))
    disp('-----Estimating test set errors------');    
   	tic; 
	[Ntest,~] = size(Xtest);
	smpidx = randperm(Ntest);
	smpidx = smpidx(1:norm_sample_size);
    Ktest = kernel(Xtest(smpidx,:),X,sigma);

    %Absolute, relative with approx computation
    [Qb,R] = qr(U,0);
    [Qs,Ls] = eig(R*diag(L)*R');
    Q = Qb*Qs;
	clear Qb Qs R
    w = Q'*Y;
    Ls = diag(Ls);
	reg = sum(Ls>lambda);
	Ls = Ls(1:reg);
	lw = w(1:reg)./Ls;
	w = Q(:,1:reg)*lw;
	
	Yguess = Ktest*w;
    absErr_approx = norm(Yguess - Ytest(smpidx))
    relErr_approx = absErr_approx/(norm(Ytest(smpidx)))
	toc
    
    %Absolute, relative with exact computation
    
else
    disp('No test set comparison available.');

end





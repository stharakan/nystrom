%%%%%  ------- %%%%%%%%
% ANY libsvm FILE - GIVE FILE UP TO _libsvm
%%%%%  ------- %%%%%%%%

%% File load parameters(if necessary)
flag = 1; % 0=data_loaded, 1=need to load
file = 'covtype_libsvm';
dir = '/org/groups/padas/lula_data/machine_learning/';
%dir ='/h2/sameer/Documents/research/nystrom/';

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
norm_sample_size = 1000;
runs = 1;
sample_method = 'random';
sigma = sigma_given(file,sigma_choice);

%% Select Lambda
disp('------Lambda Cross-Val------');
tic;
[lambda,spectrum,err] = cv_lambda(X,Y,sigma,nystrom_rank);
toc;
disp('Lamba chosen: %4.4f; Associated error: %2.5d\n', lambda, err);
%rmatvec = @(rhs) RegNystromMatVec(U,L,lambda,rhs);

%% Full Nystrom decomposition
disp('-----Nystrom decmp-------')
[N,d] = size(X);
%X = single(X);
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
%Relative
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


if ~(isempty(Xtest) || isempty(Ytest))
    disp('-----Estimating test set errors------');    
    Ktest = kernel(Xtest,X,sigma);
    

    %Absolute, relative with approx computation
    [Qb,R] = qr(U,0);
    [Qs,Ls] = eig(R*L*R');
    Q = Qb*Qs;
    w = Q*Ls*Q'*Y;
    Yguess = Ktest*w;
    absErr_approx = norm(Yguess - Ytest)
    relErr_approx = absErr_approx/(norm(Ytest))
    
    %Absolute, relative with exact computation
    
else
    disp('No test set comparison available.');

end





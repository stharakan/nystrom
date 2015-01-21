%%%%%  ------- %%%%%%%%
% ANY .askit FILE - GIVE EXACT FILE NAME
%%%%%  ------- %%%%%%%%

%% File load parameters(if necessary)
flag = 1; % 0=data_loaded, 1=need to load
file = 'covtype_scaled.askit';
dir = '/org/groups/padas/lula_data/machine_learning/';
dir ='/h2/sameer/Documents/research/nystrom/';

if flag
    clearvars -except dir file sample lambda U L
    [X,~,~,~,~] = loaddata(file,dir);
else
    clearvars -except X file sample lambda U L 
    disp(['Using previously loaded data ', file]);
end

%% Specify other parameters
nystrom_rank = 256;
sigma_choice = 3;
sflag = 1; %0=use old sample  , 1=generate new
lflag = 1; %0=lambda computed , 1=need to compute
nflag = 1; %0=nystrom computed, 1=generate new
norm_sample_size = 1000;
runs = 1;
sample_method = 'random';
sigma = sigma_given(file,sigma_choice)


%% Full Nystrom decomposition
disp('-----Nystrom decmp-------')
[N,d] = size(X);
X = single(X);
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

%% Select Lambda
disp('------Lambda Cross-Val------');
tic;
lambda = cv_lambda(X,Y,sigma,nystrom_rank);
toc;
rmatvec = @(rhs) RegNystromMatVec(U,L,lambda,rhs);

%% Estimate Errors
disp('-----Estimating kernel approx error------')
%Relative

disp('-----Estimating test set errors------');
%Absolute, relative with approx computation
%Absolute, relative with exact computation



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




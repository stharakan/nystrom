clear all;
close all;
maxNumCompThreads(1);
load ijcnn.mat;% input data matrix A should be sparse matrix with size n by d

%A=randn(1000000,16);  16D dimensionals gamma 1/2/0.45/0.45

%% ==================== parameters

k =1024; % target rank
gamma = 1; % kernel width in RBF kernel
opts.eta = 0.10000; % decide the precentage of off-diagonal blocks are set to be zero(default 0.1)
opts.noc = 10; % number of clusters(default 10)

%==================== obtain the approximation U and S(K \approx U*S*U^T)
t = cputime;
[U,S] = meka(A,k,gamma,opts); % main function
display('Done with meka!');
fprintf('The total time cost for meka is %f secs\n',cputime -t);
fprintf('***************************\n');
%==================== measure the relative error
display('Testing meka!');
[n,d] = size(A);
rsmp = 100; % sample several rows in K to measure kernel approximation error
rsmpind = randsample(1:n,rsmp);
tmpK = exp(-sqdist(A(rsmpind,:),A)*gamma);

Kapp = (U(rsmpind',:)*S)*U';

w = ones(n,1)/sqrt(n);
ex = tmpK*w;
up = Kapp*w;
Errs = norm(ex-up)/norm(ex);


Err = norm(tmpK-Kapp,'fro')/norm(tmpK,'fro');
fprintf('The relative approximation error is %.1e (fro-norm), %.1e (sample)\n',Err, Errs);

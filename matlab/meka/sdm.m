clear all;
close all;
load ijcnn.mat;% input data matrix A should be sparse matrix with size n by d

file = 'covtype';
%file = 'susy';
dir = '~/data/machine_learning/';
A=loaddata(file,dir);

%A=randn(1000000,16);  16D dimensionals gamma 1/2/0.45/0.45
%A = randn(1e6,64); % gamma = 2.62

%%
rescale = true;
save_scaled_to_file = false;

%%
%rescale data to [0,1];
if rescale
  [n,dim]=size(A);
  amin = min(A,[],2);
  amax = max(A,[],2);
  da = amax-amin;
  da = 1./da;
  nanidx = find(isinf(da)); da(nanidx)=1;   % correct for dimensions that have no variability. 
  A = A-repmat(amin,1,dim);
  A = A.*repmat(da,1,dim);
  
  % save as csv
  if save_scaled_to_file
    csvwrite([file '_scaled.askit'], full(A));
  end
end




%% ==================== parameters

k =256; % target rank
gamma = 4; % kernel width in RBF kernel
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
rsmp = 500; % sample several rows in K to measure kernel approximation error
rsmpind = randsample(1:n,rsmp);
tmpK = exp(-sqdist(A(rsmpind,:),A)*gamma);

Kapp = @(x)U(rsmpind',:)*(S*(U'*x));
%
w = ones(n,1)/sqrt(n);
ex = tmpK*w;
up = Kapp(w);
Errs = norm(ex-up)/norm(ex);


%Err = norm(tmpK-Kapp,'fro')/norm(tmpK,'fro');
fprintf('The relative approximation error is %.1e (fro-norm), %.1e (sample)\n',Errs, Errs);

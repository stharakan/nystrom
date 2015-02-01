clear all; close all;
addpath meka
%dir = '~/data/machine_learning/';
dir = '/org/groups/padas/lula_data/machine_learning/';
%dir = '';

rescale = ~true;
save_scaled_to_file = ~true;
whiten_data = ~true;

if 1
%load ijcnn.mat;% input data matrix A should be sparse matrix with size n by d
file = 'covtype_libsvm';
%file ='ijcnn';
%file = 'susy';
[A,Y,Atest,Ytest,~]=loaddata(file,dir);
%A=randn(1000000,16);  16D dimensionals gamma 1/2/0.45/0.45
%A = randn(1e6,64); % gamma = 2.62

%%
%rescale data to [0,1];
if rescale, A=scale_to_one(A); end
if whiten_data,  A=whiten(full(A),1e-8); end
if save_scaled_to_file,csvwrite([file '_scaled.askit'], full(A)); end
end

%% ==================== parameters
[n,dim]=size(A);
k =256; % target rank
%gamma = 1; H = 1/sqrt(2*gamma);
H=0.22
gamma=1/2/H/H;

opts.eta = 0.1; % decide the precentage of off-diagonal blocks are set to be zero(default 0.1)
opts.noc = 10; % number of clusters(default 10)
opts.kmeansits=15;
norm_sample_size = 1000;
runs = 1;


%rsmp = 2048; % sample several rows in K to measure kernel approximation error
%rsmpind = randsample(1:n,rsmp); 
%R =sqdist(A(rsmpind,:),A);
%num_impint  = ceil(sum(sqrt(R(:))<(4*H))/rsmp);
%tmpK = exp(-(R)*gamma);
%Kapp = @(x)U(rsmpind',:)*(S*(U'*x));
%fprintf('Interactions per point that cannot be truncated %d\n',num_impint);
%%
%==================== obtain the approximation U and S(K \approx U*S*U^T)
t = cputime;
[U_meka,S_meka] = meka(A,k,gamma,opts); % main function
display('Done with meka!');
fprintf('The total time cost for meka is %f secs\n',cputime -t);
fprintf('***************************\n');
%==================== measure the relative error
display('Testing meka!');
[n,d] = size(A);


[abs_error, rel_error] = matvec_errors(A,U_meka,S_meka,H,norm_sample_size,runs);
fprintf('The relative approximation error is %.1e (sample)\n',rel_error);


% Test regression
[U,S] = nystromorth(U_meka,S_meka);
weights = find_weights(U,S,Y,0);
[abs_err,rel_err,class_err] = regress_errors(A,Atest,Ytest, weights, sigma,norm_sample_size);
rel_err
class_err




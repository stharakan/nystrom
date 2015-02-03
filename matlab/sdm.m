clear all; close all;
addpath meka
%dir = '~/data/machine_learning/';
dir = '/org/groups/padas/lula_data/machine_learning/';
%dir = '';

rescale = ~true;
save_scaled_to_file = ~true;
whiten_data = ~true;

file = 'covtype_libsvm';
cflag = 1;
[A,Y,Atest,Ytest,~]=loaddata(file,dir);

%%
%rescale data to [0,1];
if rescale, A=scale_to_one(A); end
if whiten_data,  A=whiten(full(A),1e-8); end
if save_scaled_to_file,csvwrite([file '_scaled.askit'], full(A)); end

%% ==================== parameters
[n,dim]=size(A);
k =256; % target rank
%gamma = 1; H = 1/sqrt(2*gamma);
H=0.35
gamma=1/2/H/H;

opts.eta = 0.05; % decide the precentage of off-diagonal blocks are set to be zero(default 0.1)
opts.noc = 10; % number of clusters(default 10)
opts.kmeansits=15;
norm_sample_size = 1000;
runs = 1;

%% ==================== cross validation
[lambda,spectrum,err,errors] = cv_lambda_meka(A,Y,gamma,k,opts,cflag);
disp(['Lambda chosen: ', num2str(lambda),'; Associated error: ', num2str(err)]);
disp('-----------------------------');

%==================== obtain the approximation U and S(K \approx U*S*U^T)
disp('Computing meka approx');
t = cputime;
[U_meka,S_meka] = meka(A,k,gamma,opts); % main function
display('Done with meka!');
fprintf('The total time cost for meka is %f secs\n',cputime -t);
fprintf('-----------------------------\n');

display('Testing meka!');
[n,d] = size(A);
U_meka = full(U_meka);S_meka = full(S_meka);

tic;
%==================== measure the relative error
[abs_error, rel_error] = matvec_errors(A,U_meka,S_meka,H,norm_sample_size,runs);
disp(['The relative approximation error is ', num2str(rel_error)]);
toc

% Test regression
[U,S] = nystromorth(U_meka,S_meka);
clear U_meka S_meka
weights = find_weights(U,S,Y,lambda);

if ~(isempty(Atest) || isempty(Ytest))
	disp('Finding regression errors...');
	[abs_err,rel_err,class_corr] = regress_errors(A,Atest,Ytest, weights, H,norm_sample_size);
	rel_err
	class_corr
end




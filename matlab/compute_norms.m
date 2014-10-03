
% script to estimate the norm of the approximate kernel matrix used by the 
% Nystrom method


% Load data

N = 1000;
d = 10;

X = randn(N, d);



% pick kernel parameters

sigma = 1.0;




% create function handle

num_nystrom_samples = 10;
sample = createsample(X,num_nystrom_samples,[],'random');

[U, L] = nystromeig(X, sigma, sample);

matvec = @(rhs) NystromMatVec(U, L, rhs);

% Estimate Norms

num_norm_samples = 100;
est_norm = Estimate2Norm(matvec, num_norm_samples, N);

true_norm = norm(U*diag(L)*U');


% Output results

fprintf('Est norm: %.15f\n', est_norm);
fprintf('True norm: %.15f\n', true_norm);















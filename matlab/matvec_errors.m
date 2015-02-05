function [abs_error, rel_error] = matvec_errors(X,U,L,sigma,varargin)
if length(varargin) < 2
    norm_sample_size = 1000;
    runs = 1;
else
    norm_sample_size = varargin{1};
    runs = varargin{2};
end


[d1,d2] = size(L);
dflag = (d1 == d2);
tests = 10;
[N,r] = size(U);
norm_sample_size = min(norm_sample_size, N);
smpidx = floor(1:(N/norm_sample_size):N);

w = ones(N,1)./sqrt(N);
if dflag
    uw = L*(U'*w);
else
    uw = L.*(U'*w);
end

estKw = U(smpidx,:) * uw;
truKw = kernel(X(smpidx,:),X,sigma)*w;
N_err = norm(truKw - estKw) / norm(truKw);
disp(['Error for weights 1/sqrt(n): ', num2str(N_err)]);

abs_error = 0;
rel_error = 0;

for j = 1:tests
    w = normrnd(0,1,[N,1]);
    w = w./norm(w);
    if dflag
        uw = L*(U'*w);
    else
        uw = L.*(U'*w);
    end
    
    
    if(runs ~=1)
        abs_error = 0;
        denom = 0;
        newN = norm_sample_size/runs;
        for i = 1:runs
            sidx = smpidx(((i-1)*newN+1):i*newN);
            estKw = U(sidx,:) *uw;
            truKw = kernel(X(sidx,:),X,sigma)*w;
            abs_error = abs_error + norm(estKw - truKw)^2; %/norm_sample_size;
            denom = denom + norm(truKw)^2;
        end
        rel_error = sqrt(rel_error/denom);
    else
        estKw = U(smpidx,:) * uw;
        truKw = kernel(X(smpidx,:),X,sigma)*w;
        err = norm(truKw - estKw);
        abs_error = abs_error + err;
        rel_error = rel_error + (err / norm(truKw)); %/norm_sample_size;
    end
end
abs_error = abs_error / tests;
rel_error = rel_error / tests;

end

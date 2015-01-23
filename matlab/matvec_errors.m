function [abs_error, rel_error] = matvec_errors(X,U,L,norm_sample_size,runs)

if ~exist('norm_sample_size')
	norm_sample_size = 1000;
end
if ~exist('runs')
	runs = 1;
end

[N,r] = size(U);
norm_sample_size = min(norm_sample_size, N);

smpidx = randperm(N);
smpidx = smpidx(1:norm_sample_size);
w = ones(N,1)./sqrt(N);
uw = L.*(U'*w);

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
    abs_error = norm(truKw - estKw);
	rel_error = abs_error/ norm(truKw); %/norm_sample_size;
end


end

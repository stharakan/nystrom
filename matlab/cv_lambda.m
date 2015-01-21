function [ lambda,errors ] = cv_lambda(X,Y,sigma,nystrom_rank)
%CV_LAMBDA Runs 5-fold cross-validation to find a suitable lambda. Lambdas
%are chosen based on the estimated spectrum generated by the Nystrom
%approximation. 

%% Set initial values
FOLDS = 5;
[N,d] = size(X);
nystrom_m = nystrom_rank*2;
sflag = 1; %determines whether we need to subsample in each section
if nystrom_rank > 4 * N/FOLDS
    sflag = 0;
end
    
%% Split into sections
fold_sizes = zeros(FOLDS,1);
bigs = mod(N,FOLDS);
fold_sizes(1:bigs) = ceil(N/FOLDS);
fold_sizes(bigs+1:end) = floor(N/FOLDS);
shuffled = randperm(N);
sample = 1:fold_sizes(1) - fold_sizes(1);


%% Specify lambda choices
num_gams = 1024;
curr_sample = createsample(X,num_gams*2,[],'random');
[~, spectrum] = nystromeig(X,sigma,curr_sample,num_gams,0);
spectrum = spectrum(end:-1:1);


%% Choose ideal lambda
errors = zeros(size(spectrum));

for i = 1:FOLDS
	disp(['Working fold ', num2str(i)]);
    sample = sample + fold_sizes(i);
    curr_idx = diff(1:N,sample);
    curr_idx = shuffled(curr_idx);
    
    curr_sample = createsample(X(curr_idx,:),nystrom_m,[],'random');
    disp('Running nystrom...');
	[U, L] = nystromeig(X(curr_idx,:),sigma,curr_sample,nystrom_rank,1);
    
    [Qb,R] = qr(U,0);
    [Qs,D] = eig(R*diag(L)*R');
    D = diag(D);
    Q = Qb*Qs;
    
    trainy = Y(curr_idx);
    qy = (Q'*y)./D;
    
    Ktest = kernel(X(sample, :), X(curr_idx,:), sigma);
    testy = Y(sample);
        
    dcount = length(D);
    old_err = 0;
    disp('Recording errors...');
    for j = 1:length(spectrum)
        %Figure out last value to include
        if D(dcount) < spectrum(j)
            while D(dcount) < spectrum(j)
                dcount = dcount - 1;
                if dcount == 0, break;
                end;
            end
            
            esty = Ktest*Q(:,1:dcount)*qy(1:dcount);
            old_err = norm(esty-testy)
            errors(j) = errors(j) + old_err;
        else
            errors(j) = errors(j) + old_err;
        end
        
    end

end

errors = errors./FOLDS;
lambda = min(errors);



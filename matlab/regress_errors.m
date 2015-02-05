function [relErr_approx,class_err] = regress_errors(X,Xtest,Ytest,w,sigma,varargin)
%REGRESS_ERRORS computes regression errors on the test set Xtest, given a
%a sigma, weights w, and the correct labels Ytest
approx = 0;
ll = length(varargin);
str = 'exact';
if ll > 2
    Um = varargin{1};
    Un = varargin{2};
    Ln = varargin{3};
    w = Um * (Ln.*(Un'*w));
    all_flag = 0;
    approx = 1;
    str = 'approx';
    
    if ll == 4
        all_flag = varargin{4};
    elseif ll >4  
        disp('Incorrect format for regresserrors.m');
    end
end

absErr_approx = 0;
relErr_approx = 0;
class_err = 0;
[N,d] = size(X);
[Ntest,~] = size(Xtest);
norm_sample_size = 1000;

smpidx = floor(1:(Ntest/norm_sample_size):Ntest);

if approx
    if all_flag
        smpidx = 1:Ntest;
    else
        smpidx = floor(1:(Ntest/norm_sample_size):Ntest);
    end

    if (N * Ntest > 10E9)
        disp('Make sure SAMPLE X is entered, matrices too large. Exiting..');
        return;
    end
end

Ktest = kernel(Xtest(smpidx,:),X,sigma);

Yguess = Ktest*w;
absErr_approx = norm(Yguess - Ytest(smpidx));
relErr_approx = absErr_approx/(norm(Ytest(smpidx)));
class_err = sum( sign(Ytest(smpidx)) == sign(Yguess) )/length(smpidx);
disp(['Error tested on ', num2str(length(smpidx)),' points', ...
    ' with the ', str, ' kernel']);
end

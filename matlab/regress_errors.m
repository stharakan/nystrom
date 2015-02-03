function [absErr_approx, relErr_approx,class_err] = regress_errors(X,Xtest,Ytest,w,sigma,varargin)
%REGRESS_ERRORS computes regression errors on the test set Xtest, given a
%a sigma, weights w, and the correct labels Ytest
all_flag = 0;
if length(varargin) < 1
    norm_sample_size = 1000;
else
    [d1,d2] = size(varargin{1});
    if d1~=d2
        Um = varargin{2};
        w = Um * w;
        all_flag = 1;
    else
        norm_sample_size = varargin{1};
    end
end

absErr_approx = 0;
relErr_approx = 0;
class_err = 0;
[N,d] = size(X);
[Ntest,~] = size(Xtest);

if ~all_flag
    smpidx = floor(1:Ntest/norm_sample_size:Ntest);
else
    disp('Testing entire set with Nystrom approx.');
    smpidx = 1:Ntest;
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
end


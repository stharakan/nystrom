function [absErr_approx, relErr_approx] = regress_errors(Xtest,Ytest,w,sigma,norm_sample_size)
%REGRESS_ERRORS computes regression errors on the test set Xtest, given a
%a sigma, weights w, and the correct labels Ytest
if exist('norm_sample_size')
    norm_sample_size = 1000;
end

[Ntest,~] = size(Xtest);
smpidx = randperm(Ntest);
smpidx = smpidx(1:norm_sample_size);
Ktest = kernel(Xtest(smpidx,:),X,sigma);

Yguess = Ktest*w;
absErr_approx = norm(Yguess - Ytest(smpidx));
relErr_approx = absErr_approx/(norm(Ytest(smpidx)));

end

function [ w ] = gen_kde_weights(Ytest)
%GEN_KDE_WEIGHTS Generates the kde weights used based on a binary
%classifier

w = Ytest;
w(w == 1) = 1/sum(w== 1);
w(w ==-1) = -1/sum(w==-1);
w = w./norm(w);


end


function [w] = find_weights(Q,D,Y,lambda)
%FIND_WEIGHTS Finds the weights for the given decomposition and Y value by
%inverting the matrix. Solves Kx = Y, with K = Q*D*Q'
n = length(D);
d = D(D>lambda);
reg = length(d);

qy = Q'*Y;
qy = qy(1:reg)./d;

w = Q(:,1:reg)*qy;


end


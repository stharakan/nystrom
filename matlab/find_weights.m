function [varargout] = find_weights(Q,D,Y,lambda)
%FIND_WEIGHTS Finds the weights for the given decomposition and Y value by
%inverting the matrix. Solves Kx = Y, with K = Q*D*Q'

d = D(D>lambda);
reg = length(d);

qy = Q'*Y;
qy = qy(1:reg)./d);

w = Q(:,1:reg)*qy;

varargout{1} = w;
varargout{2} = qy;

end


function w = find_weights(Q,D,Y,lambda)
%FIND_WEIGHTS Finds the weights for the given decomposition and Y value by
%inverting the matrix. Solves Kx = Y, with K = Q*D*Q'

d = D(D>lambda);
reg = length(d);

qy = Q'*Y;

w = Q(:,1:reg)*(qy(1:reg)./d);


end


function [ v ] = NystromMatVec( U,L, rhs )
% simple function handle for nystrom approximation U*L*U'

    v = U*diag(L)*U'*rhs;

end


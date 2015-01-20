function [ out ] = RegNystromMatVec( U,L, lambda,rhs )
% simple function handle for nystrom approximation U*L*U'
ur = U'*rhs;
ur = ur.*L;
v = U*ur;

out = v + lambda.*rhs;
end


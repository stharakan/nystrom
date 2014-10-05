function [ v ] = NystromMatVec( U,L, rhs )
% simple function handle for nystrom approximation U*L*U'
				ur = U'*rhs;
				ur = ur.*L;
				v = U*ur;

end


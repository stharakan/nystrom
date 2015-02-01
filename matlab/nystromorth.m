function [Q, L] = nystromorth(Un,Ln)
%NYSTROMORTH orthogonalizes the matrices output by nystromeig
%
%   INPUTS:
%       - Un: "orthogonal" system generated by nystrom
%       - Ln: vector of diagonal eigenvalues
%       
%   OUTPUTS:
%       - Q:  orthogonalized version of Un
%       - L: vector of adapted eigenvalues
[d1,d2] = size(Ln);
dflag = d1 == d2;

[Qb,R] = qr(Un,0);
clear Un

if dflag
	[Qs,L] = svd(R*Ln*R');
else
	[Qs,L] = svd(R*diag(Ln)*R');
end
clear R Ln

L = diag(L);
Q = Qb*Qs;
clear Qb Qs

end

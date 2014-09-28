function [a_norms, residuals, spectra] = makeLcurve(X,Y,m,sigmas,sample)
%MAKELCURVE creates an L-curve for each value of sigma, comparing
%the norm of the residual with the norm of alpha. 

%Initialize
numsigs = length(sigmas);
[n,~] = size(X);
a_norms = zeros(m,numsigs);
residuals = zeros(m,numsigs);

spectra=zeros(m,numsigs);
%Loop through each sigma

for ii = 1:numsigs
	disp(['Generating L curve for sigma ', num2str(ii)]);
	sig = sigmas(ii);

	%Decompose K into K = U L U'
	[U,L] = nystromeig(X,sig,sample);
	
	%Orthogonalize U = Q R
	[Q,R] = qr(U,0);
	
	%Eigendecompose inner matrix R L R' = Qs S Qs', diagonalize
	[Qs,S] = eig(R*diag(L)*R');
	S = diag(S);
  spectra(:,ii) = S;
  
% 	disp(['Highest eval of s: ', num2str(S(1))]);
% 	disp(['Lowest eval of s : ', num2str(S(end))]);

	%Project Y onto subspace spanned by Q * Qs
	qy = Qs'*(Q'*Y);

	%Solve for rhs
 	s_qy = qy./S;
	
	%Compute the minimum possible residual (i.e. dist(Y, span(Q * Qs)))
	base_res = norm(Y - Q*Qs*qy,2)^2;

	%Corresponds to using jj eigenvalues of S to solve
	for jj = 1:m
		%||alpha_jj|| = ||Qs_jj' Q' alpha_jj|| = ||S_jj^-1 Qs_jj' Q' Y ||
		a_norms(jj,ii) = norm(s_qy(1:jj),2);
		
		%||K alpha_jj - Y|| = sqrt(base error^2 + in_space error^2) 
		residuals(jj,ii) = sqrt(base_res + norm(qy(jj+1:end),2)^2);
	end
end
disp('All curves made');

end

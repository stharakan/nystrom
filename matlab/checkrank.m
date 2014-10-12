
use_plot = ~true;
%P=A'; 
%%
[dim,N]=size(P);
M=1000;
Q=P(:,unique(randi(N,M,1)));

%P = randn(dim,N); % set of points (with mean slightly shifted by some factor of HS
%Q = randn(dim,M);

%%
SC = 0.5;  % SCALING of Silverman BW
H = SC*silverman(dim,N);
H=0.16;
fprintf('Actual bandwidth is %g\n',H);
gaussian = @(r) exp(-1/2 * r.^2);
rho = 2;



% % pick a random point and create a sphere of rho h around it, rho<1 and create approximation. 
P0 = Q(:,randi(M,1)); 
R0 = distance(P,P0)/H;
nidx = find( R0 <= rho);
fidx = find( R0 > rho);
fprintf('%d points cannot be truncated\n',length(nidx));
fprintf('relative near (<rho), relative far(>rho) magnitude %.2e %.2e\n',...
   norm(gaussian(R0(nidx)))/norm(gaussian(R0)), norm(gaussian(R0(fidx)))/norm(gaussian(R0)));

%%
RQ = distance(P,Q);

%%
H=0.16;
fprintf('size/H of the box is %.1e\n', max(RQ(:))/2/H);
GQ = gaussian(RQ/H);
sq=svd(GQ,'econ');
sq = sq/max(sq);
fprintf('Rank of %d X %d matrix (sum(s>9e-3) = %d\n', N,M,sum(sq>9e-3));


%%
if use_plot
  GQ=GQ(:);
  subplot(3,1,1),hist(RQ(:),10);
  subplot(3,1,2),hist(log(GQ(RQ(:)<6)),100);
  subplot(3,1,3),semilogy(sq);
end

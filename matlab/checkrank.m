clear all; clear globals; 
N = 100000; % 
M = 1000; % 
dim = 10;      % dimension
rescale  = true;
loadfile = true;

%dir = '/org/groups/padas/lula_data/machine_learning/';
dir = '~/data/machine_learning/';
file = {'covtype','susy','mnist2m_scaled.askit','ijcnn.askit','mnist8m_scaled_nocommas.askit'};
if loadfile, P=single(loaddata(file{4},dir)); end;
if rescale, P=scale_to_one(P); end

[N,dim]=size(P); 
Q=P(unique(randi(N,M,1)),:);
%P = randn(dim,N); 
%Q = randn(dim,M);
%%
H = SC*silverman(dim,N);
fprintf('Actual bandwidth is %g\n',H);
H=4;
fprintf('Bandwidth H= %g, SC=%1.2f\n',H,SC);
gaussian = @(r) exp(-1/2 * r.^2);
rho = 4;

P0 = Q(:,randi(M,1)); 
R0 = distance(P,P0)/H;
nidx = find( R0 <= rho);
fidx = find( R0 > rho);
fprintf('%d points cannot be truncated\n',length(nidx));
fprintf('relative near (<rho), relative far(>rho) magnitude %.2e %.2e\n',...
   norm(gaussian(R0(nidx)))/norm(gaussian(R0)), norm(gaussian(R0(fidx)))/norm(gaussian(R0)));
hidx = find( (R0(:)<=2)&(R0(:)>=0.5) );
fprintf('Num points in the high derivative region=%d\n',length(hidx));

Nr = 1e5;
ri=randi(N,Nr,1);
RQ = distance(P(ri,:)',Q')/H;
GQ = gaussian(RQ);
sq=svd(GQ,'econ');
sq = sq/max(sq);
fprintf('Rank of %d X %d matrix (sum(s>9e-3) = %d\n', N,M,sum(sq>9e-3));

[bins,values]=hist(GQ(:));

%GQ=GQ(:);

%%
use_plot = ~true;
if use_plot
  GQ=GQ(:);
  subplot(3,1,1),hist(RQ(:),10);
  subplot(3,1,2),hist(log(GQ(RQ(:)<6)),100);
  subplot(3,1,3),semilogy(sq);
end

clear GQ;

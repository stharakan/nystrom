% This scripts checks the distribution of the partition between far field and near field for 
% a random set of target points, given a bandwidth and a near-far field splitting critierion.


if 0
  clear all; clear globals; 
  gaussian = @(r) exp(-1/2 * r.^2);
  gbloaddata;
  P=P';
 [N,dim]=size(P);
end



rounds =500;
P0 = P(:,randi(N,rounds,1)); 
R0_orig = distance(P0,P);

%% near - far field criterion. If distance/H < rho, then it is near field.
cutoff = 3;  
H=0.10;
R0=R0_orig/H;

for jj=1:rounds
  nidx = find( R0(jj,:) <= cutoff);
  fidx = find( R0(jj,:) > cutoff);
  hidx = find( (R0(jj,:)<=2)&(R0(jj,:)>=0.5) );
  total(jj) = norm(gaussian(R0(jj,:)));
  near(jj) =norm(gaussian(R0(jj,nidx)))/total(jj);
  far(jj)  =norm(gaussian(R0(jj,fidx)))/total(jj);
  len_nidx(jj) = length(nidx);
  len_hidx(jj) = length(hidx);
end
%
fprintf('\n------- Statistics-sigma=%1.2f---cutoff=%1d sigma---- (median,max,min,std)\n',H,cutoff);
fprintf('near count: %7d   %7d  %7d   %8.1f\n',  uint32(median(len_nidx)), max(len_nidx), min(len_nidx), std(len_nidx));
fprintf('derv count: %7d   %7d  %7d   %8.1f\n',  uint32(median(len_hidx)), max(len_hidx), min(len_hidx), std(len_hidx));
fprintf('near field: %1.2e  %1.2e  %1.2e  %1.2e\n',  median(near), max(near), min(near), std(near));
fprintf('far field : %1.2e  %1.2e  %1.2e  %1.2e\n',  median(far), max(far), min(far), std(far));
fprintf('\n');

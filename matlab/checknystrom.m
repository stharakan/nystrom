gaussian=@(r)exp(-1/2 * r.^2);

%% check rank of dataset using randomized sampling. 
loadfile = true;
dir='~/data/machine_learning/';
file = {'covtype_libsvm','susy','mnist2m_scaled.askit','ijcnn.askit','mnist8m_scaled_nocommas.askit'};
if loadfile, P=loaddata(file{1},dir); end;
%%
[N,dim] = size(P);

% subsample points from dataset
ell = 1*1024;  % sample size

sample_ids = unique(randi(N,ell,1)); ell = length(sample_ids);

Ps = P(sample_ids,:);
R = distance(Ps',Ps');
H=0.10;
K = gaussian(R/H);
s = N/ell * svd(K);

%
%s3=s;
%%
%clf;semilogy(s3,'r','LineWidth',8);grid on; hold on; semilogy(s2,'b','LineWidth',4); semilogy(s1,'c'); 
%title('H=0.10','FontSize',22);

%% here select a large group of points, split them in half, and test the ranks of the diagonal and off-diagonal blocks;

ell = 4096; sample_ids = unique(randi(N,ell,1)); ell = length(sample_ids);
Ps = P(sample_ids,:);
idx = kmeans(Ps,2);

%%
P1=Ps(idx==1,:);
P2=Ps(idx==2,:);
H=0.11;
K11=gaussian(distance(P1',P1')/H);
K12=gaussian(distance(P1',P2')/H);
%%
s11=svd(K11);
s12=svd(K12);
hold off;semilogy(s11/max(s11)); hold on; semilogy(s12/max(s12),'r');

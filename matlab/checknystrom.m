%% DEFINE kernel
clear all; clear globals; clf;
gaussian=@(r)exp(-1/2 * r.^2);
visualize = true;

%% check rank of dataset using randomized sampling. 
loadfile = true;
dir='~/data/machine_learning/';
file = {'covtype_libsvm','susy','mnist2m_scaled_nocommas.askit','ijcnn.askit','mnist8m_scaled_nocommas.askit'};
if loadfile, P=loaddata(file{1},dir); end;
%%
[N,dim] = size(P);

% here select a large group of points, split them in half, and test the ranks of the diagonal and off-diagonal blocks;

% number of sampling points
ell = 4024; 
sample_ids = unique(randi(N,ell,1)); 
ell = length(sample_ids);
Ps = P(sample_ids,:);
idx = kmeans(Ps,2);
fprintf('number of points per cluster: cluster 1: %5d, cluster 2:%5d\n', sum(idx==1), sum(idx==2));
%%
H=0.22; % kernel bandwidth
P1=Ps(idx==1,:);
P2=Ps(idx==2,:);

K11=gaussian(distance(P1',P1')/H);
K21=gaussian(distance(P2',P1')/H);
%%
s11=svd(K11);
s21=svd(K21);

if visualize, hold off;semilogy(s11/max(s11)); hold on; semilogy(s21/max(s11),'r'); end;

%% Study effecti of nearest neighbors.
% find nearest neigbhors of P1 in P2, Remove them from the list of the far
% targets; that is remove them for the list of targets for which
% approximation takes place. 
% remove 1...k interactions from K(P2,P1) and see the effect on the w,
% means removing raws from K.
%
% let's simulate our algorithms. 

k = 16;
nn = direct_knn(Ps',Ps',32);  % find k  nearest neighbors using exact search
%%
P1nn = nn(idx==1,:);     %  P1nn: all neighbors of points in the P1 set
P1nn = unique(P1nn(:));

% neighbors of P1 in P2
n12 = P1nn(idx(P1nn)==2);
fprintf('Set P1 has %d neighbors in set P2\n', length(n12));
allidx = [1:size(Ps,1)]';
P2idx = allidx(idx==2);

% keep_askit: all remaining points in P2. That is, points in P2 that are
% _not_ neighbors of points in P1
[~,keep_askit] = setdiff(P2idx,n12);
%%
K21k=K21(keep_askit,:);
s21k=svd(K21k);
if visualize, semilogy(s21k/max(s11),'y','LineWidth',3); end;

sidx= [1,100,min(1024,length(s21k))]
[s11(sidx),s21(sidx),s21k(sidx)]/max(s11)











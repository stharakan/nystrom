%% DEFINE kernel
clear all; clear globals; clf;
addpath ~/projects/knn/matlab/bintree/
split_type_mtree = -1;split_kmeans     = 2;

%%
gaussian=@(r)exp(-1/2 * r.^2);
visualize =~true;

%% check rank of dataset using randomized sampling. 
loadfile = true;
dir='~/data/machine_learning/';
%file = {'covtype_libsvm','susy','mnist2m_scaled_nocommas.askit','ijcnn.askit','mnist8m_scaled_nocommas.askit'};
%if loadfile, P=loaddata(file{1},dir); end;
load([dir,'covtype.mat']);
%%
[N,dim] = size(P);

% here select a large group of points, split them in half, and test the ranks of the diagonal and off-diagonal blocks;

% number of sampling points
ell = 4096*4; 
sample_ids = randperm(N); 
sample_ids = sample_ids(1:ell);
Ps = P(sample_ids,:);
split_type = split_kmeans;
split_type = split_type_mtree;
idx = splitpoints(Ps',split_type);
%idx = kmeans(Ps,2);
fprintf('number of points per cluster: cluster 1: %5d, cluster 2:%5d\n', sum(idx==1), sum(idx==2));
%%
H=0.08; % kernel bandwidth
P1=Ps(idx==1,:);
P2=Ps(idx==2,:);

K11=gaussian(distance(P1',P1')/H);
K21=gaussian(distance(P2',P1')/H);
%s=svd( gaussian(distance(Ps',Ps')/H) )*N/ell;
%%
s11=svd(K11);
s21=svd(K21);

% if visualize, hold off;semilogy(s11/max(s11)); hold on; semilogy(s21/max(s11),'r'); end;
[s11([1,128,1024,2048]), s21([1,128,1024,2048])]/max(s11)

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











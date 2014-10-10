function [idx] = kmeanspick(X,m,ll)
%KMEANSPICK Picks out m points using kmeans clustering, according to how
%many points are in each class
%
%   INPUTS:
%       - X: n x d matrix of n points sitting in d-dim feature space. 
%       - m: size of the system to be solved exactly 
%       - ll: vector of length classes (INCLUDING UNLABELED) giving the 
%       number of points in each class
%
%   OUTPUTS:
%       - idx: list of the m columns of X to use to generate the
%       eigendecomposition of SVD decomp

[n,~] = size(X);
idx = zeros(m,1); %idx of points chosen by kmeans
num_classes = length(ll); %number of classes present

%Determine number of points to pick from each class by scaling. If off,
%pick one at random
clustsperclass  = round(ll.*(m/n));
diff = sum(clustsperclass) - m;
if (diff ~= 0)
    fixer = [sign(-diff)*ones(length(diff),1); zeros(num_classes-length(diff),1)];
    fixer = fixer(randperm(num_classes));
    clustsperclass = clustsperclass + fixer;
end

%Actually compute kmeans on each cluster
labcount = 0;
for ii = 1:num_classes
    [clusters,~,~,dists] = kmeans(X(labcount+1:labcount+ll(ii),:), ...
        clustsperclass(ii),'EmptyAction','singleton','MaxIter',20);
    count = sum(clustsperclass(1:ii-1));
    for jj = 1:clustsperclass(ii)
        [~,within_clust_idx] = min(dists(clusters == jj, jj));
        clust_idx = find(clusters == jj);
        idx(count+jj) = labcount + clust_idx(within_clust_idx);
    end
    labcount = labcount + ll(ii);
end

end


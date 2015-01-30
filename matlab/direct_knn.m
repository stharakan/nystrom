function [idk,idr,dists] =  direct_knn (query,reference,k_neighbors,range)
% computes the pairwise L2 (not squared) distance matrix between any column
% vectors in X and in x
%
% INPUT:
% reference     dxN matrix consisting of N column vectors for reference points
% query        dxn matrix consisting of m column for query points
%
% OUTPUT:
% idk: k-nn ids
% idr: range-nn ids
% dists: distances for k-nn neighbors.


D=distance(query,reference);
[Ds,in]=sort(D,2, 'ascend');

idk   = in(:,1:k_neighbors);
dists = D;

m = size(query,2);
idr{m} = [];
if nargin>4
  for j=1:m, idr{j} = find(D(j,:) <= range); end;
end








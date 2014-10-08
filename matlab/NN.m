function [idx,sigma] = NN(Data,points,k,flag)
%NN Given a dataset, assorted points and a number k, outputs indices of k
%nearest neighbors of each point in points, as well as a sigma range.
%
%   Note that this calculation requires the number of features to be of a
%   small dimension (i.e. <10) since it utilizes kdtrees. The indices of
%   idx can be used to formulate the resulting matrices , and the min max
%   values are for the sigma parameter in the kernel and weight functions.
%   
%   INPUTS:
%       - Data: n x d matrix of n points sitting in d-dim feature space.
%       - points: m x d matrix of m points sitting in d-dim feature space
%       - k: number of nearest neighbors to be computed
%       - flag: if true, this says that points are a subset of Data, and
%       thus the first nearest neighbor is irrelevant. if false, points are
%       not a subset of Data, and then the first nearest neighbor is
%       important
%       
%   OUTPUTS:
%       - idx:  m x k matrix of indices listing the indices corresponding to
%       the k nearest elements of Data to any element of points
%       - sigma: vector list of range of possible sigma values based on the
%       distances to the last neighbor.

kdtreeNS = KDTreeSearcher(Data);

if(flag) % points is contained in Data
    [idx,dist] = knnsearch(kdtreeNS, points, 'k', k+1);
    idx = idx(:,2:k+1);
    dist = dist(:,2:k+1);
else % points not in Data
    [idx,dist] = knnsearch(kdtreeNS, points, 'k', k);
end

% figure
% plot(dist(:,end))
% hold on
% plot(dist(:,1),'r')
% xlabel('index')
% ylabel('distance')
% title('Distances to furthest vs nearest neighbor')
% legend('Furthest neighbor','Nearest neighbor')

sigma = linspace(max(dist(:,1))/2, 40*max(dist(:,end)), 10);
sigma = sigma./4;


end


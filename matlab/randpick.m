function [idx] = randpick(m,n)
%RANDPICK Picks the m kernel columns to be used in a completely random
%fashion
%   Detailed explanation goes here

idx = datasample(1:n, m, 'Replace', false);
idx = sort(idx);
end


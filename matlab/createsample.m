function idx = createsample(X,m,Y,method)
[n,d]=size(X);

if n==m, idx = 1:m'; return; end;

if strcmp(method,'random')
    idx = randpick(m,n);
%kmeans
elseif strcmp(method,'kmeans')
    idx = kmeanspick(X,m,n);

else strcmp(method, 'kmeans with classes');
   ll = histc(Y,unique(Y));
   idx = kmeanspick(X,m,ll);
end

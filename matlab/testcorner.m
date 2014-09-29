% create some random matrix K with decaying eigenvalues:
n=100; noiselevel=1e-1;
[Q,~] = qr(randn(n));
L=[n:-1:1]'./n^2;
K=Q*diag(L)*Q';
aex = rand(n,1);
noise = norm(aex,inf)* noiselevel *rand(n,1);
f = K*aex + noise;

a = (Q'*f)./L;

eta = cumsum(abs(a));
rho = flipud(cumsum(abs(f)));

clf
hold on;
loglog(eta,rho);
gamma=corner(eta,rho)
loglog(eta(gamma), rho(gamma), '.', 'MarkerSize',30); grid on;
hold off;


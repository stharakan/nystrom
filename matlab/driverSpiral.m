% LSDRIVER is a script that runs all the necessary functions relating to
% the regularized least algorithm. First it creates a dataset through
% create_set, then finds an optimal system by running NN and findsigma. It
% then finds an appropriate gammaA and gammaI, reevaluating the classifier
% each time. 

clear all
close all

filename = 'spiral_cs8000_n05';
k = 6; % # of nearest neighbors
labeled = 4000; % # of labeled points in each class 
m = 256; % size of smaller system to solve
cv_folds = 5; % # of cross-val folds

disp('Loading training data ... ')
load(['train',filename]);
[n,~] = size(X); n = n/2;
idx = [1:labeled n+1:n+labeled labeled+1:n n+labeled+1:2*n];

%%
Xtrain = X(idx,:);
Ytrain = zeros(size(Y));
Ytrain(1:2*labeled) = Y(idx(1:2*labeled));
disp('Loading test data ... ')
load(['test',filename]);
Xtest = X;
Yexact = Y;
disp('Done loading data!')
clear X Y;


%% Find NN to get sigma range
disp('Entering NN code ... ' )
[idx,sigmas] = NN(Xtrain,Xtrain,k,1);
disp('NNs calculated. Finding sigma ... ')
warning('off','all');
warning

%%
%sigmas=sigmas(1:2);
disp('Generating L curve data ...');
sample = createsample(Xtrain,m,Ytrain,'kmeans with classes');
[alpha_norms, res_norms, spectra] = makeLcurve(Xtrain, Ytrain, m, sigmas,sample);

%%
corners = zeros(size(sigmas));  % L-curve analysis
gcv     = zeros(size(sigmas));       % generalized cross validation
mpr   = zeros(size(sigmas));     % minimum product alphas.*res hack.

gcv_d=[1:m]'; gcv_d=m-gcv_d;
fprintf('----------------------------------------------------------------------------\n');
fprintf('        kc   rc \t ac\t  | kg   rg\t  ag \t   |km \t   rm\t\t am\t | rmin \n');
fprintf('----------------------------------------------------------------------------\n');
for i= 1:length(sigmas)
  % Lcurve analysis
	[k,info] = corner(res_norms(:,i),alpha_norms(:,i));
	corners(i) = k;
  corner_best(i) = res_norms(k,i);
  corner_k(i) = k;
  
  % generalized cross-validation with truncated SVD
  [~,kg] = min(res_norms(:,i)./gcv_d);
  gcv(i)=res_norms(kg,i);
  gcv_best(i) = res_norms(kg,i);
  gcv_k(i) = kg;
  
  % hack
  [~,km] = min(res_norms(:,i).*alpha_norms(:,i));
  mpr(i) = res_norms(km,i);
  mpr_best(i) = res_norms(km,i);
  mpr_k(i) = km;
  
  
  fprintf('[i=%2d] %03d %.1e %.1e| %03d %.1e %.1e| %03d %.1e %.1e | %.1e\n',...,
  i,k, res_norms(k,i), alpha_norms(k,i),...
    kg,res_norms(kg,i),alpha_norms(kg,i),...
    km,res_norms(km,i),alpha_norms(km,i),...
    res_norms(end,i));
  
  
  hold off; 
  loglog(res_norms(:,i),alpha_norms(:,i));
  hold on;
  loglog(res_norms(k,i),alpha_norms(k,i),'.','MarkerSize',10);
  loglog(res_norms(kg,i),alpha_norms(kg,i),'.r','MarkerSize',10);
  loglog(res_norms(km,i),alpha_norms(km,i),'.g','MarkerSize',10);
  hold off;
  pause(1);
end
[~,imin] = min(corner_best); sigma_c = sigmas(imin); gamma_c= spectra(corner_k(imin),imin);
[~,imin] = min(gcv_best);    sigma_g = sigmas(imin); gamma_g= spectra(gcv_k(imin),imin);
[~,imin] = min(mpr_best);    sigma_m = sigmas(imin); gamma_m= spectra(mpr_k(imin),imin);

sigma_win= [sigma_c,sigma_g,sigma_m];
gamma_win= [gamma_c, gamma_g, gamma_m]; 

%%
%disp('Saving L curve data ... ');
%save([filename,'_Lcurves'], 'alpha_norms','res_norms','sigmas');


%Find optimal sigma from given range
%[sigma,Us,Ls,alpha] = findsigma(Xtrain,Ytrain,m,sigmas,cv_folds);

for win=1:3
  sigma = sigma_win(win);
  gamma = gamma_win(win);
  alpha = nystromsolve(Xtrain,Ytrain,sample,sigma, gamma);

  %Evaluate error after calculating sigma
  ystar = testdata(alpha,sigma,Xtrain,Xtest);
  correctclass = sum(ystar(1:n) > 0) + sum(ystar(n+1:end) < 0);
  disp(['Correctly classified ', num2str(correctclass), ' out of ', ...
    num2str(2*n), ' elements'])
end
%%
%Find regularizer gamma
%[gammaA,alpha,knorms,residuals] = findgammaA(Xtrain,Ytrain,sigma,m,cv_folds);

%Evaluate error after calculating gamma
%ystar = testdata(alpha,sigma,Xtrain,Xtest);
%correctclass = sum(ystar(1:n) > 0) + sum(ystar(n+1:end) < 0);
%disp(['Correctly classified ', num2str(correctclass), ' out of ', ...
%    num2str(2*n), ' elements'])


%{
Ycalc = testdata(alpha,sigma,Xtrain,Xtest);
figure
plot(1:length(Ycalc),Ycalc,1:length(Ycalc), 0)
legend('Calculated labels','Dividing line (y = 0)')
xlabel('index')
ylabel('class')
%title('Classification after determining sigma')

%solve for gammaA
[gammaA,alpha] = findgammaA(Xtrain,Ytrain,Ksig,sigma);
Ycalc = testdata(alpha,sigma,Xtrain,Xtest);

figure
plot(1:length(Ycalc),Ycalc,1:length(Ycalc), 0)
legend('Calculated labels','Dividing line (y = 0)')
xlabel('index')
ylabel('class')
%title('Classification after determining gammaA')

%solve for gammaI
[gammaI,alpha] = findgammaI(Xtrain,Ytrain,Ksig,sigma,gammaA);
Ycalc = testdata(alpha,sigma,Xtrain,Xtest);

figure
plot(1:length(Ycalc),Ycalc,1:length(Ycalc), 0)
legend('Calculated labels','Dividing line (y = 0)')
xlabel('index')
ylabel('f value')
%title('Classification after determining gammaI')


temp = sum(Xtrain.*Xtrain,2)';

f = @(x,y) (exp(-(repmat(x.^2+y.^2,1,n*2)+temp-2*[x,y]*Xtrain')/(2*sigma^2)))*alpha;
figure
hold on
ezcontourf(f, [-1,1])
%title('Countour plot of f')
colorbar
hold on
plot(Xtest(1:n,1),Xtest(1:n,2),'xw',Xtest(n+1:end,1),Xtest(n+1:end,2),'.w')
%}

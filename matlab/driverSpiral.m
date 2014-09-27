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
m = 1024; % size of smaller system to solve
cv_folds = 5; % # of cross-val folds

disp('Loading training data ... ')
load(['train',filename]);
[n,~] = size(X); n = n/2;
idx = [1:labeled n+1:n+labeled labeled+1:n n+labeled+1:2*n];
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

disp('Generating L curve data ...');
[alpha_norms, res_norms] = makeLcurve(Xtrain, Ytrain, m, sigmas);
alpha_norms = log(alpha_norms);
res_norms = log(res_norms);

%%
%corners = zeros(size(sigmas));
%for i= 1:length(sigmas)
%	[k,info] = corner(res_norms(:,i),alpha_norms(:,i));
%	corners(i) = k;
%end

disp('Saving L curve data ... ');
save([filename,'_Lcurves'], 'alpha_norms','res_norms','sigmas');

%Find optimal sigma from given range
%[sigma,Us,Ls,alpha] = findsigma(Xtrain,Ytrain,m,sigmas,cv_folds);

%Evaluate error after calculating sigma
%ystar = testdata(alpha,sigma,Xtrain,Xtest);
%correctclass = sum(ystar(1:n) > 0) + sum(ystar(n+1:end) < 0);
%disp(['Correctly classified ', num2str(correctclass), ' out of ', ...
%    num2str(2*n), ' elements'])

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

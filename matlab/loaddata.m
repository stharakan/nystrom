function [Xtrain,Ytrain,Xtest,Yexact,n] = loaddata(dataset,labeled)
%LOADDATA will output the training and testdata set given which file to load
% under the following specifications
% 'spiral' - test/trainspiral_cs8000_n05
% 'wine'   - winequality-red/white.csv

if ~exist('labeled') %ignored for non-synthetic datasets
				labeled = 0;
end


if isequal(dataset, 'spiral')
				filename = 'spiral_cs8000_n05';
				disp('Loading training data ... ')
				load(['train',filename]);
				[n,~] = size(X); n = n/2;

				if ~exist('labeled') %ignored for non-synthetic datasets
								disp('Setting labeled to n/2')
								labeled = n;
				end

				idx = [1:labeled n+1:n+labeled labeled+1:n n+labeled+1:2*n];

				Xtrain = X(idx,:);
				Ytrain = zeros(size(Y));
				Ytrain(1:2*labeled) = Y(idx(1:2*labeled));
				disp('Loading test data ... ')
				load(['test',filename]);
				Xtest = X;
				Yexact = Y;
				disp('Done loading data!')

elseif isequal(dataset,'wine')
				disp('Reading in red wine data')
				data_red = dlmread('/scratch/data/machine_learning/winequality-red.csv',';',1,0);

				disp('Reading in white wine data')
				data_white = dlmread('/scratch/data/machine_learning/winequality-white.csv',';',1,0);

				Xtrain = [data_red(:,1:end-1) ; data_white(:,1:end-1)];
				Xtest = Xtrain;
				Ytrain = [data_red(:,end); data_white(:,end)];
				Yexact = Ytrain;
				disp('Done loading data!')
				[n, ~] = size(Xtrain);
else 
				disp('Error: Data not found - type help loaddata for options');
end





end

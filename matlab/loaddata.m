function [Xtrain,Ytrain,Xtest,Yexact,n] = loaddata(dataset,dir,labeled)
%LOADDATA will output the training and testdata set given which file to load
% under the following specifications
% 'spiral' - test/trainspiral_cs8000_n05
% 'wine'   - winequality-red/white.csv

if ~exist('labeled') %ignored for non-synthetic datasets
				labeled = 0;
end

addpath(dir);

if isequal(dataset, 'spiral')
				filename = 'spiral_cs8000_n05';
				disp('Loading training data ... ')
				load([dir,'train',filename]);
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
				load([dir,'test',filename]);
				Xtest = X;
				Yexact = Y;
				disp('Done loading data!')

elseif isequal(dataset,'wine')
				
				disp('Reading in red wine data')
				filename = 'winequality-';
				data_red = dlmread([dir,filename,'red.csv'],';',1,0);

				disp('Reading in white wine data')
				data_white = dlmread([dir,filename,'white.csv'],';',1,0);

				Xtrain = [data_red(:,1:end-1) ; data_white(:,1:end-1)];
				Xtest = [];
				Ytrain = [data_red(:,end); data_white(:,end)];
				Yexact = [];
				[n, ~] = size(Xtrain);
				disp('Done loading data! (Test and training are the same)')

elseif isequal(dataset,'susy')
				
				disp('Reading in SUSY data')
				filename = 'SUSY.csv';
				x = load([dir,filename]);
				
				Ytrain = x(:,1);
				Yexact = [];
				Xtrain = x(:,2:end);
				Xtest = [];
				[n,~] = size(Xtrain);
				disp('Done loading data! (Test and training are the same)')

elseif isequal(dataset,'cpu')

				disp('Reading in cpusmall data');
				filename = 'cpusmall';
				fid = fopen([dir,filename]);
				x = textscan(fid,'%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f');
				fclose(fid);
				idx = 1 + 2.*(1:12);
				
				Ytrain = x{1}; Yexact = [];
				Xtrain = zeros(length(x{1}),12);
				Xtest = [];
				for i = 1:length(idx)
								Xtrain(:,i) = x{idx(i)};
				end
				n = length(x{1});
				disp('Done loading data! (Test and training are the same)');

elseif isequal(dataset,'covtype')
				
				disp('Reading in covertype data')
				filename = 'covtype.data';
				x = csvread(filename);
				Xtrain = x(:,1:end-1);
				[n,~] = size(Xtrain);
				Ytrain = x(:,end);
				Yexact = []; Xtest = [];
				disp('Done loading data! (Test and training are the same)');

else
				disp('Error: Data not found - type help loaddata for options');
end





end

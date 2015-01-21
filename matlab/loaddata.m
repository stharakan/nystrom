function [Xtrain,Ytrain,Xtest,Yexact,n] = loaddata(dataset,dir,labeled)
%LOADDATA will output the training and testdata set given which file to load
% under the following specifications
% 'spiral' - test/trainspiral_cs8000_n05
% 'wine'   - winequality-red/white.csv
% 'susy' - unscaled susy data
% 'covtype' - unscaled covtype
% 'ijcnn1' - unscaled ijcnn1
% 'cpusmall' - unscaled cpusmall
%
%%%%% -------- %%%%%%%%
% ANY .askit FILE - GIVE EXACT FILE NAME
%%%%% -------- %%%%%%%%
%%%%% -------- %%%%%%%%
% ICML datasets pass only part of name up to _libsvm
%%%%% -------- %%%%%%%%


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
    
elseif isequal(dataset,'covtype')
    
    disp('Reading in covertype data')
    filename = 'covtype.data';
    x = csvread(filename);
    Xtrain = x(:,1:end-1);
    [n,~] = size(Xtrain);
    Ytrain = x(:,end);
    Yexact = []; Xtest = [];
    disp('Done loading data! (Test and training are the same)');
    
elseif isequal(dataset,'ijcnn1') || isequal(dataset,'cpusmall')
    
    disp(['Reading in ', dataset, ' data'])
    fid = fopen([dir,dataset]);
    x = textscan(fid,'','Delimiter',{':',' '});
    Ytrain = x{1};
    
    n = length(Ytrain); m = max(x{end-1});
    Xtrain = zeros(n, m);
    for i = 1: (length(x)-1)/2
        Xtrain = Xtrain + sparse(1:n,x{2*i},x{2*i + 1},n,m);
    end
    Xtest = []; Yexact = [];
    fclose(fid);
    
elseif sum([repmat(' ',1,length(dataset) -7), '_libsvm'] == dataset) == 6
    disp(['Reading in ', dataset]);
    Xtrain = load([dir,dataset,'_train_askit']);
    Ytrain = load([dir,dataset,'_train_askit_labels']);
    Xtest = load([dir,dataset,'_test_askit']);
    Yexact = load([dir,dataset,'_test_askit_labels']);
    [n1, ~] = size(Xtrain); [n2, ~] = size(Xtest);
    n = n1+n2;
    disp('Done loading data! (Training and test loaded)');

elseif strncmp(dataset,'gauss',5) || strncmp(dataset,'hyper',5) || sum([repmat(' ',1,length(dataset)-6), '.askit'] == dataset) == 6    
    disp(['Reading in ', dataset]);
    Xtrain = load([dir,dataset]);
    Ytrain = []; Yexact = []; Xtest = [];
    [n,~] = size(Xtrain);
    
    disp('Done loading data! (Test and training are the same, no Y)')
else
    disp('Error: Data not found - type help loaddata for options');
end





end

function sigma = sigma_given(dataset)
% Outputs the sigma given in the MEKA paper for the following datasets (otherwise use Silverman)
%
% cpusmall - 'cpusmall'
% winequality - 'wine'
% SUSY - 'susy' 
% covertype - 'covtype'


if isequal(dataset,'cpusmall')
				sigma = sqrt(1/8);
elseif isequal(dataset,'wine')
				sigma = sqrt(2^9);
elseif isequal(dataset,'susy')
				sigma = 1.0;
elseif isequal(dataset,'covtype')
				sigma = sqrt(1/8);
elseif isequal(dataset,'ijcnn1')
				sigma = sqrt(1/2);
elseif isequal(dataset,'hypersphere_4d_100K.txt')
				sigma = 0.2143;
elseif isequal(dataset,'hypersphere_4d_1M.txt')
				sigma = 0.1607;
elseif isequal(dataset,'gaussian_16d_100K.txt')
				sigma = 0.5060;
elseif isequal(dataset,'gaussian_16d_1M.txt')
				sigma = 0.4510;
elseif isequal(dataset,'gaussian_32d_100K.txt')
				sigma = 0.6722;
elseif isequal(dataset,'gaussian_32d_1M.txt')
				sigma = 0.6305;
else
				disp('File not recognized')
end

function sigma = sigma_given(dataset)
% Outputs the sigma given in the MEKA paper for the following datasets (otherwise use Silverman)
%
% cpusmall - 'cpusmall'
% winequality - 'wine'
% SUSY - 'susy' 
% covertype - 'covtype'
% susy_scaled - 'susysc'
% ijcnn1 - 'ijcnn1'
% Synthetic data -- use actual file name

if isequal(dataset,'cpusmall')
				sigma = sqrt(1/8);
elseif isequal(dataset,'wine')
				sigma = sqrt(2^9);
elseif strncmp(dataset,'susy',4)
				sigma = 1.0;
elseif strncmp(dataset,'covtype',7);
				sigma = sqrt(1/8);
elseif strncmp(dataset,'ijcnn',5)
				sigma = sqrt(1/2);
elseif isequal(dataset,'mnist2m_scaled_nocommas.askit')
				sigma = 4.0;
elseif isequal(dataset,'mnist8m_scaled_nocommas.askit')
				sigma = 3.99;
elseif isequal(dataset,'hypersphere_4d_100K.askit')
				sigma = 0.2143;
elseif isequal(dataset,'hypersphere_4d_1M.askit')
				sigma = 0.1607;
elseif isequal(dataset,'gaussian_16d_100K.askit')
				sigma = 0.5060;
elseif isequal(dataset,'gaussian_16d_1M.askit')
				sigma = 0.4510;
elseif isequal(dataset,'gaussian_32d_100K.askit')
				sigma = 0.6722;
elseif isequal(dataset,'gaussian_32d_1M.askit')
				sigma = 0.6305;
else
				disp('File not recognized')
end

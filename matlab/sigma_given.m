function sigma = sigma_given(dataset,flag)
% Outputs the sigma given in the MEKA paper for the following datasets (otherwise use Silverman)
% --> FOR MULTIPLE SIGMAS FLAG INDICATES CHOICE (1 ,2 .. = largest -> smallest)
%
% cpusmall - 'cpusmall'
% winequality - 'wine'
% SUSY - 'susy' 
% covertype - 'covtype'
% susy_scaled - 'susysc'
% ijcnn1 - 'ijcnn1'
% Synthetic data -- use actual file name
if ~exist('flag')
    flag = 1;
end

if isequal(dataset,'cpusmall')
    sigma = sqrt(1/8);
elseif isequal(dataset,'wine')
    sigma = sqrt(2^9);
elseif strncmp(dataset,'susy',4)
    if flag == 1, sigma = 1.0;
    elseif flag == 2, sigma = 0.3;
    elseif flag == 3, sigma = 0.15;
    elseif flag == 4, sigma = 0.1;
    elseif flag == 5, sigma = 0.07;
    end
elseif strncmp(dataset,'covtype',7);
    if flag == 1, sigma = 0.35;
    elseif flag ==2, sigma = .22;
    elseif flag ==3, sigma = .16;
    end
elseif strncmp(dataset,'ijcnn',5)
    if flag == 1, sigma = .7071;
    elseif flag == 2,sigma = 0.35;
    end
elseif strncmp(dataset,'mnist2m',7)
    if flag ==1, sigma = 4.0;
    elseif flag ==2, sigma =3.0;
    elseif flag ==3, sigma =2.0;
    end
elseif strncmp(dataset,'mnist8m',7)
    sigma = 4;
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

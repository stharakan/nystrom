% Script to test the matrix multiplies at different sigmas:
% 0.35 --> takes everything to 8th power
% 0.22 --> takes everything to 20th power
% 0.16 --> takes everything to 40th power
m = 512;
n = 58100;

% Create random matrix with entries between 0 and 1
test = rand(m, n);
test = test.^8;
vec = rand(n,1);
svec = single(vec);
stest = single(test);

% Specify which powers (8, 20, 40)
powers =[1,20/8,40/8];

% Evaluate multiplies
for i=1:length(powers)
	pow = powers(i);
	disp(['Currently working with power ', num2str(pow)]);
	curr_test = test.^pow;
	scurr_test = stest.^pow;
	
	%time double multiply
	tic;
	curr_test*vec;
	d_time = toc;
	disp(['Double multiply took ',num2str(d_time)]);
	
	%time single multiply
	tic;
	scurr_test*svec;
	s_time = toc;
	disp(['Single multiply took ', num2str(s_time)]);

end

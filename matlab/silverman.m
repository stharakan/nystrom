function bandwidth = silverman(dim,num_of_points);

D = dim;
N = num_of_points;

c1 = 1/(D+4); AK=(4/(D+2))^c1;
HS =  AK/(N^c1);  

bandwidth = HS;


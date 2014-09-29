function potential = kernel(targets, sources,sigma)
% function potential = kernel(targets, sources,sigma)
% exp( - rho.^2 ./ (2*sigma.^2 );
% potential function  

rho = distance(targets',sources');
potential = exp( - rho.^2 ./ (2*sigma.^2 ));

end

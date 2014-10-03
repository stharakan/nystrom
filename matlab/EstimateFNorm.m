function [ result ] = EstimateFNorm( matvec_handle, num_samples, N )

% Estimates F-norm of the matrix
% applies to randomly selected standard basis vectors to extract 
% columns, then computes their 2-norm

    result = 0;

    for i = 1:num_samples
       
        id = randi(N, 1);
        u = zeros(N,1);
        u(id) = 1;
        
        v = matvec_handle(u);
        
        result = result + v'*v;
        
    end
    
    result = sqrt(result * N / num_samples);

end




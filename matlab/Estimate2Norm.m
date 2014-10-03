function [result] = Estimate2Norm(matvec_handle, num_samples, N)

% Estimates 2-norm of the matrix with the power method
% Does num_samples iterations
% N - size of the matrix
% matvec_handle -- performs the matrix vector multiply

    u = randn(N,1);
    u = u / norm(u);
   
    for i=1:num_samples

        u = matvec_handle(u);
        u = u / norm(u);
        
    end
 
    u = matvec_handle(u);
    result = norm(u);
    
end
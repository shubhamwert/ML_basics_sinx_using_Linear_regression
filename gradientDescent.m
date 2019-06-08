function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); 
J_history = zeros(num_iters, 1);
   

for iter = 1:num_iters

	cal=(X*theta-y).*X;
	
theta=theta-(alpha*sum(cal)/m)';


    % ============================================================

    % Save the cost J in every iteration    

    J_history(iter)=cost(X, y, theta);
   
    

end

end

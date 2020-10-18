function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%fprintf('\ntheta %d %d\n', size(theta)(1,1), size(theta)(1,2)); % 4x1
%fprintf('X %d %d\n', size(X)(1,1), size(X)(1,2)); % 5x4
%fprintf('y %d %d\n', size(y)(1,1), size(y)(1,2)); % 5x1
%fprintf('size(theta,1) %d\n', size(theta,1)); 
%fprintf('size(theta)(1,1) %d\n', size(theta)(1,1)); 


A = log(sigmoid(theta' * X')) * (-1) * y;
B = log(1 - sigmoid(theta' * X')) * (1 - y);
C = (lambda/(2 * m)) * sum(theta(2:size(theta,1), 1) .^ 2);

J  = (1/m) * (A - B) + C;

G = sigmoid(theta' * X') - y';
grad(1,:) = (1/m) * G * X(:,1);

L = (lambda/m) * theta(2:size(theta,1), 1); 
grad(2:size(grad,1), :) = (1/m) * G * X(:, 2:size(X,2)) + L';


% =============================================================

grad = grad(:);

end

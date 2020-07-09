function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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
for i = 1:m 
    cos = (1/m) * ((-y(i)) * log(sigmoid(theta' * X(i,:)')) - (1-y(i)) * log(1 - sigmoid(theta' * X(i,:)')));
    J = J + cos;  
endfor

for j = 2:size(theta)
    penalty = (lambda/(2*m)) * theta(j)^2;
    J = J + penalty;
endfor 

% method one:
for j = 1:size(theta)
    for i = 1:m 
        gradient = 1/m * (sigmoid(theta' * X(i,:)') - y(i)) * X(:,j)(i);
        grad(j) = grad(j) + gradient;
    endfor
        
    if j == 1
        grad(j) = grad(j);
    else
        grad(j) = grad(j) + (lambda/m) * theta(j);
    endif
endfor



% method two:
% for i = 1:m 
%     gradient = 1/m * (sigmoid(theta' * X(i,:)') - y(i)) * X(:,1)(i);
%     grad(1) = grad(1) + gradient;
% endfor

% for j = 2:size(theta)
%     for i = 1:m 
%         gradient = 1/m * (sigmoid(theta' * X(i,:)') - y(i)) * X(:,j)(i);
%         grad(j) = grad(j) + gradient;
%     endfor
%     grad(j) = grad(j) + (lambda/m) * theta(j);
% endfor




% =============================================================

end

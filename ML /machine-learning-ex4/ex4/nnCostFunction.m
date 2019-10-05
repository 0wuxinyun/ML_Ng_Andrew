function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% part 1 & 3:  feedpropogation GET J(0)-COST:


ynew=zeros(num_labels,m);
for i=1:m
ynew(y(i),i)=1;

end
% X->a:
X=[ones(m,1),X];
a=sigmoid(X*Theta1');
Theta11=Theta1(:,2:end);
regtheta1=Theta11(:);
% a->z:
a=[ones(m,1),a];
z=sigmoid(a*Theta2');  %m*num -> z'=num*m
Theta22=Theta2(:,2:end);
regtheta2=Theta22(:);
% z->J:
ynew1=ynew(:);  %1,2,3,:

z=z';

z1=z(:); 
e=[regtheta1;regtheta2];
J=-1/m*(log(z1)'*ynew1+log(1-z1)'*(1-ynew1))+lambda/(2*m)*(e'*e);


%part2:  backpropogation:
% sigmaL:  theta2(num_label*n+1)  theta1:(n,input+1) input()=

sigma1=z-ynew; % num_label*m
sigma2=((Theta2)'*sigma1).*(a'.*(1-a)'); %(n+1*m)
%sigma3=((Theta2)'*sigma2(2:end,:)).*X.*(1-X);
%delta:
Theta1_grad(:,1)=1/m*(sigma2(2:end,:)*X(:,1));

Theta1_grad(:,2:end)=(1/m*(sigma2(2:end,:)*X(:,2:end)))+(lambda./m.*Theta1(:,2:end));%element-wise

Theta2_grad(:,1)=1/m*(sigma1*a(:,1));

Theta2_grad(:,2:end)=(1/m*(sigma1*a(:,2:end)))+(lambda./m.*Theta2(:,2:end));%element-wise












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

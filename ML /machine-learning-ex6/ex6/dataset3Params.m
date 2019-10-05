function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
% initial width
width=[0.01;0.03;0.1;0.3;1;3;10;30];


indices=zeros(length(width),length(width));
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% traing sets ->get parameters
for i=1:length(width)  %c
    for j=1:length(width) %sigma
        C=width(i);
        sigma=width(j);

% modelL?
         model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));  
          prediction=svmPredict(model, Xval);

% accuracy:
          indices(i,j)=mean(double(prediction~=yval));

    end
end

% minrow=min(indices);
%minone=min(minrow);
%n=find(minrow==minone);% column number
%mincolumn=indices(:,n);
%m=find(mincolumn==minone);
%C=width(i);
%sigma=width(j);

%

error=min(min(indices));
[i,j]=find(indices==error);
C=width(i);
sigma=width(j);




% =========================================================================

end

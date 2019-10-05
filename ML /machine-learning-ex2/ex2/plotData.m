function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

figure;
% classification the dataset:
pass=find(y==1);
fail=find(y==0);
%  it seems one graph HOWEVER: it`s more precisely TWO graphs:
plot(X(pass,1),X(pass,2),'ko','MarkerSize',7,'color','r');
hold on;
plot(X(fail,1),X(fail,2),'k+','MarkerSize',7);






% =========================================================================



hold off;

end

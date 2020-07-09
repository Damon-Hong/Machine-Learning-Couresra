function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

positiveIdx = find(y == 1);
negativeIdx = find(y == 0);

plot(X(positiveIdx, 1), X(positiveIdx, 2), "k+")
plot(X(negativeIdx, 1), X(negativeIdx, 2), "ko")
% xlabel("Exam 1 score")
% ylabel("Exam 2 score")
% legend("Admitted", "Not admitted")






% =========================================================================



hold off;

end

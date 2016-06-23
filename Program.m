clear; close all; clc;

% read the training data
data = load('ex1data1.txt');

% initialize Matrices and Variables
X = data(:, 1);     % featue matrix
y = data(:, 2);     % results matrix
m = length(y);      % number of training examples
theta = zeros(2, 1);     % initial weights
iterations = 1500;  % Iterations needed for Gradient Descent
alpha = 0.01;       % Learning Rate

% Plot the Data
plot(X, y, 'rx', 'MarkerSize', 10);
title('Training Examples');
xlabel('Population in 10,000');
ylabel('Profit in $10,000');

% Compute the Cost Function
X = [ones(m, 1), data(:, 1)];
J = ComputeCost(X, y, theta);

% Run Gradient Descent
[theta, Js] = GradientDescent(X, y, theta, alpha, iterations);
hold on;
plot(X(:, 2), X * theta, '-');
legend('Training data', 'Linear regression');
hold off;

% Visualize The Cost Function
fprintf('Visualizing J(theta_0, theta_1) ...\n');

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = ComputeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

% Predicting Prices
fprintf('Prediction for 35000:\t%f\n', ([1, 3.5] * theta) * 10000);
fprintf('Prediction for 70000:\t%f\n', ([1, 7] * theta) * 10000);
function J = ComputeCost(X, y, theta)
    % Prepare Variables
    m = length(y);
    
    % Calculate Hypothesis
    h = X * theta;
    
    % Calculate Cost
    J = 1 / (2 * m) * sum((h - y) .^ 2);
end
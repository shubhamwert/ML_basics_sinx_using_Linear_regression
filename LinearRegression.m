
X=zeros(100,1);
for num=1:100
    X(num)=0.001*pi+(num-1)*0.01;


end
fprintf('Value of X:\n');

Y=sin(X)


hold off;
plot(X, Y, 'rx', 'MarkerSize', 5);
X=[X];
theta=zeros(2,1);

X = [ones(100, 1), X];
alpha=0.01;
iterations=5000;


J=lrcost(X,Y,theta);
[theta,JK] = lrgradientDescent(X, Y, theta, alpha, iterations);
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);
fprintf('lrcost found at this theta\n');
fprintf('%f\n', lrcost(X,Y,theta));



% Plot

hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

fprintf('accuracy: %f\n',sum(sqrt((X*theta-Y).^2))/length(Y));
fprintf('cost  for new data\n');
SX=[0.3;0.4;0.5;0.6;1];


SX=[ones(length(SX),1),SX]
costp=lrcost(SX,sin(SX),theta)

fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = lrcost(X, Y, t);
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

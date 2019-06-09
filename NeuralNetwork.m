clear ; close all; clc

%Controll training parameters




%data%
X=zeros(6000,1);
for num=1:6000
    X(num)=-0.0050*pi+0.001*num*pi+sqrt(num)*0.00001;


end

Y=sin(X);


hold off;
plot(X, Y, 'rx', 'MarkerSize', 5);
X=[X,X.^3,cos(X)];


%NN parameters%
input_layer_size = 3;
hidden_layer_size=8;
lables=2;


m = size(X, 1);

Theta1=rand(hidden_layer_size,input_layer_size+1);
Theta2=rand(lables,hidden_layer_size+1);
nn_params = [Theta1(:) ; Theta2(:)];
pause;
%forward Feeding of NN%

lambda=0;
fprintf('calculating J')
J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size, lables,X, Y, lambda);

%regularization%
lambda = 1;
J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size, lables,X, Y, lambda);


%inital parameters



fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(hidden_layer_size,input_layer_size);
initial_Theta2 = randInitializeWeights(lables,hidden_layer_size);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
J = nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size, lables,X, Y, lambda)

%check if nn is fine%
checkNNGradients;

lambda = 3;
checkNNGradients(lambda);

%NN training
fprintf('\nTraining Neural Network... \n')
iter=4000;

options = optimset('MaxIter', iter);
lambda = 1;


% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   lables, X, Y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 lables, (hidden_layer_size + 1));


testX=[0;-pi/2;pi;pi/4;-pi/4;pi/3;7*pi/4]
y_matrix=sin(testX)>0
y_matrix=y_matrix+1;
testX=[testX,testX.^3,cos(testX)];
sin(testX)
result=predict(Theta1,Theta2,testX)
testX=zeros(1000,1);
for num=1:1000
    testX(num)=-0.50*pi+0.001*num*pi+sqrt(num)*0.0001;


end
y=sin(testX);

testX=[ones(size(testX),1),testX,testX.^3,cos(testX)];
result=predict(Theta1,Theta2,testX);
nn_params = [Theta1(:) ; Theta2(:)];

J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size, lables,testX, y, lambda)
fprintf('1 -----> positive\n 2 -------> negative');
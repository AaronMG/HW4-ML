function prob1
    loc = 'C://Users/gonza647/CS/Courses/CSE847-Machine Learning/HW4/';
    cd(loc)
    
    data = importdata('data.txt', ' ', 0);      %4601 x 57
    labels = importdata('labels.txt', ' ', 0);  %4601 x 1
    data = [data ones(4601,1)];                 %Add the intercept term to data
    epsilon = 0;                             
    maxiter = 0;                             
    
    % Create the train and test sets
    x_train = data(1:2000,:);
    y_train = labels(1:2000,:);
    x_test = data(2001:end,:);
    y_test = labels(2001:end,:);
    
    n = [200 500 800 1000 1500 2000];
    accuracy = [];
    for i=1:length(n)
        correct = 0;
        n_xtrain = x_train(1:n(i),:);
        n_ytrain = y_train(1:n(i),:);
        weights = logistic_train(n_xtrain, n_ytrain, epsilon, maxiter);
        prediction = sigmf(x_test*weights, [1 0]);
        pred = round(prediction);
        for j = 1:length(y_test)
           if y_test(j) == pred(j)
               correct = correct + 1;
           end
        end
        accuracy(i) = correct/length(y_test);
        fprintf('n = %d\tAccuracy = %f\n', n(i), accuracy(i));
    end
    plot(n, accuracy);
    
end

function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
% INPUTS:
%   data    = n * (d+1) matrix with n samples and d features, where column
%             d+1 is all ones (corresponding to the intercept term)
%   labels  = n * 1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence criterion - if
%             the change in the absolute difference in predictions, from one
%             iteration to the next, averaged across input features, is less than
%             epsilon, then half (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the max number of iterations
%             to execute. (If unspecified can be set to 1000)
%
% OUTPUT:
%       weights = (d+1) * l vector of weights where the weights correspond
%       to the columns of "data"
    if ~maxiter
        maxiter = 1000;
    end
    if ~epsilon
        epsilon = 1e-5;
    end

    weights = zeros(58,1);                      %Initialize all weights     

    N = length(data);
    R = zeros(N,N);
    for i = 1:maxiter
        y = sigmf(data*weights, [1 0]);        
        for n = 1:N
           R(n,n) = y(n)*(1-y(n)); 
        end    
        wOld = weights;
        R = R + eye(N)*0.1;
        %Compute gradient and Hessian matrix for cross-entropy error func
        z = data*wOld - inv(R)*(y-labels);
        weights = inv(transpose(data)*R*data)*transpose(data)*R*z;
    end
end
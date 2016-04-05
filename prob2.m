clear;
clc;
load('ad_data.mat', 'X_train', 'y_train', 'X_test', 'y_test');

%Set the options as they were set in the assignment example
opts.rFlag = 1;
opts.tol = 1e-6;
opts.tFlag = 4;
opts.maxIter = 5000;

%Use regularization parameters ranging from 0 to 1, with 0.05 intervals
par = [0:0.01:1];
acc = [];
perf = [];
for i =1 :length(par)           % Test every parameter
    [w,c] = LogisticR(X_train, y_train, par(i), opts);  %Train data using LogisticR
    pred = X_test*w + c;        % Calculate our predictions, use c as the bias
    [~,~,~,perf(i)] = perfcurve(y_test, pred, 1);    %Take necessary AUC information
end
% Plot the AUC curve 
figure; 
plot(par,perf);
title('AUC Curve');

clear;
clc;
load('ad_data.mat', 'X_train', 'y_train', 'X_test', 'y_test');

opts.rFlag = 1;
opts.tol = 1e-6;
opts.tFlag = 4;
opts.maxIter = 5000;
par = [0:0.05:1];
acc = [];
perf = [];
for i =1 :length(par)
    [w,c] = LogisticR(X_train, y_train, par(i), opts);
    right = 0;
    pred = X_test*w + c;
    [~,~,~,perf(i)] = perfcurve(y_test, pred, 1);    
    for j=1:length(pred)
       if pred(j) < 0 
           pred(j) = -1;
       else 
           pred(j) = 1;
       end
       if pred(j) == y_test(j)
          right = right + 1;
       end
    end
    acc(i) = right/length(pred);
end
figure;
plot(par,perf);
figure;
plot(par, acc);

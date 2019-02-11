
clear all;close all;

% Generating the data
x = [randn(20,2);randn(20,2)+4];
y = [repmat(-1,20,1);ones(20,1)];

% Adding a bad point :)
x = [x;2 1];
y = [y;1];
% Plotting the data
types = {'ko','ks'};
fc = {[0 0 0],[1 1 1]};
val = unique(y);
ind = find(y==val(1));


figure(1); hold off
for i = 1:length(val)
    ind = find(y==val(i));
    plot(x(ind,1),x(ind,2),types{i},'markerfacecolor',fc{i});
       hold on
end

% Setting up the optimization problem
N = size(x,1);
K = x*x';
H = (y*y').*K + 1e-5*eye(N);
f = ones(N,1);
A = [];b = [];
LB = zeros(N,1); UB = inf(N,1);
Aeq = y';beq = 0;


warning off


%Different values of Regularization parameters
Cvals = [10 5 2 1 0.5 0.1 0.05 0.01]; 

for cv = 1:length(Cvals)
    UB = repmat(Cvals(cv),N,1);
    % Following line runs the SVM
    alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB);
    % Compute the bias
    fout = sum(repmat(alpha.*y,1,N).*K,1)';
    ind = find(alpha>1e-6);
    bias = mean(y(ind)-fout(ind));
    
    %Plot the data, decision boundary and Support vectors
    figure; hold off
    ind = find(alpha>1e-6);
    plot(x(ind,1),x(ind,2),'ko','markersize',15,'markerfacecolor',[0.6 0.6 0.6],...
        'markeredgecolor',[0.6 0.6 0.6]);
    hold on
    for i = 1:length(val)
        ind = find(y==val(i));
        plot(x(ind,1),x(ind,2),types{i},'markerfacecolor',fc{i});
    end

    xp = xlim;
    yl = ylim;
    % Because this is a linear SVM, we can compute w and plot the decision
    % boundary exactly.
    w = sum(repmat(alpha.*y,1,2).*x,1)';
    yp = -(bias + w(1)*xp)/w(2);
    plot(xp,yp,'k','linewidth',2);
    ylim(yl);
    ti = sprintf('C: %g',Cvals(cv));
    title(ti);
end
%In the end you will get different figures with different regularization
%parameters. You can see that as Cvals is increasing the strictness of
%our line is increasing. If Cvals tend to infinity then we will get
%hard margin classifier.
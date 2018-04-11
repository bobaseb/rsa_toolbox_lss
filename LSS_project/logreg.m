function [acc] = logreg(X_train,Y_train,X_test,Y_test)
%logreg gives accuracy of multinomial logistic regression, taking off
%columns if training matrix is ill-behaved

try
    [B,~,~] = mnrfit(X_train,Y_train');
catch %if matrix is not PD, then take off features
    X_train = X_train(:,1:end-1);
    X_test = X_test(:,1:end-1);
    [B,~,~] = mnrfit(X_train,Y_train');
end
pihat = mnrval(B,X_test);
preds=[];
for i = 1:length(Y_test)
    [~,pred] = max(pihat(i,:));
    preds(i) = pred;
end

acc = sum(preds==Y_test)./length(Y_test);

end


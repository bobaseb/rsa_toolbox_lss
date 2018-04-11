function [anovas] = anova_vec(X_train,Y_train)
%anova_vec Vectorized ANOVAs

nconditions = length(unique(Y_train));

bg_var = zeros(1,length(X_train));
wg_var =  zeros(1,length(X_train));
grand_means = mean(X_train);
for cond = 1:nconditions
    msk = Y_train==cond;
    sample_means = mean(X_train(msk,:));
    bg_var = bg_var + sum(msk)*((sample_means-grand_means).^2)/(nconditions-1); %between-group variance
    samples = X_train(msk,:);
    wg_var = wg_var + sum((samples-sample_means).^2)/(length(Y_train)-nconditions); %within-group variance
end
anovas = bg_var./wg_var;

end


function mean_acc = run_clf(glm_model,DATA,nconditions,nruns,nf,svm_options)
%% evaluate glm models
runs = 1:nruns;

for run = 1:nruns
    
    X_test =  zscore(DATA.(sprintf('run%d',run)).fMRI.(sprintf(glm_model)));
    Y_test =  DATA.(sprintf('run%d',run)).fMRI.sequence;
    Y_test = Y_test(Y_test~=nconditions+1); %takes out null trials
    
    train_runs = runs~=run;
    
    Y_train=[];
    X_train=[];
    for train_run = 1:nruns
        if train_runs(train_run)
            X_train =  [X_train; zscore(DATA.(sprintf('run%d',train_run)).fMRI.(sprintf(glm_model)))];
            Y_train_tmp = DATA.(sprintf('run%d',train_run)).fMRI.sequence;
            Y_train_tmp = Y_train_tmp(Y_train_tmp~=nconditions+1); %takes out null trials
            Y_train =  [Y_train Y_train_tmp];
        end
    end
    
    %% do feature selection (anovas)
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
    [~,voxel_ranks] = sort(anovas,'descend');
    
    chosen_voxels = voxel_ranks(1:nf);
    
     X_test = X_test(:,chosen_voxels);
     %X_test = zscore(X_test); zscore up top
     X_train = X_train(:,chosen_voxels);
     %X_train = zscore(X_train); zscore up top
    %% run classifiers
    
    %[B,~,~] = mnrfit(X_train,Y_train');
    %X_train
    %pihat = mnrval(B,X_test);
    %preds=[];
    %for i = 1:length(Y_test)
    %    [~,pred] = max(pihat(i,:));
    %    preds(i) = pred;
    %end
    %preds
    %(preds==Y_test)
    %accs(run) = sum(preds==Y_test)./length(Y_test);
    
    svm_model = svmtrain(Y_train', X_train, svm_options); %nu-svm is supposed to be -s 1??? -t 0 is linear kernel, -t 2 rbf
    [~, acc, ~] = svmpredict(Y_test', X_test, svm_model);
    accs(run) = acc(1);
end

mean_acc = mean(accs);
end
function [all_chosen_voxels, mean_acc] = run_clf(glm_model,DATA,nconditions,nruns,nf,svm_options)
%runs CV classifiers (SVM or logistic regression), needs anova_vec function for feature selection

%% evaluate glm models
runs = 1:nruns;

all_chosen_voxels=[];
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
    
    %% do feature selection (anovas/single voxel)
    if nf==1
        X_test = X_test(:,DATA.rand_signal_voxel);
        X_train = X_train(:,DATA.rand_signal_voxel);
    else
        anovas = anova_vec(X_train,Y_train);
        [~,voxel_ranks] = sort(anovas,'descend');
        chosen_voxels = voxel_ranks(1:nf);
        all_chosen_voxels = [all_chosen_voxels; chosen_voxels];
        X_test = X_test(:,chosen_voxels);
        X_train = X_train(:,chosen_voxels);
    end
    %% run classifiers
    
    if svm_options==0
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
        accs(run) = sum(preds==Y_test)./length(Y_test);
    else
        svm_model = svmtrain(Y_train', X_train, svm_options); %nu-svm is supposed to be -s 1??? -t 0 is linear kernel, -t 2 rbf
        [~, acc, ~] = svmpredict(Y_test', X_test, svm_model);
        accs(run) = acc(1);
    end
    
end

mean_acc = mean(accs);
end
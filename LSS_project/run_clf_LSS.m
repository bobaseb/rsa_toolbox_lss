function [all_chosen_voxels, mean_acc] = run_clf_LSS(glm_model,DATA,nconditions,nruns,nf,svm_options)
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
    if nf==1 || nf==-1
        if nf==-1 %create neighbour average, do before single voxel
            X_test2 = mean(X_test(:,DATA.rsv_cube_inds),2);
            X_train2 = mean(X_train(:,DATA.rsv_cube_inds),2);
        end
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
        acc = logreg(X_train,Y_train,X_test,Y_test);
        accs(run) = acc;
    else
        svm_model = svmtrain(Y_train', X_train, svm_options); % -t 0 is linear kernel, -t 2 rbf
        [~, acc, ~] = svmpredict(Y_test', X_test, svm_model);
        accs(run) = acc(1);
    end
    
    %run classifier also for average from neighbours if nf==-1
    if nf==-1 & svm_options==0
        acc2 = logreg(X_train2,Y_train,X_test2,Y_test);
        accs2(run) = acc2;
    elseif nf==-1
        svm_model2 = svmtrain(Y_train', X_train2, svm_options); % -t 0 is linear kernel, -t 2 rbf
        [~, acc2, ~] = svmpredict(Y_test', X_test2, svm_model2);
        accs2(run) = acc2(1);
    end
    
end

mean_acc = mean(accs);

if nf==-1
    mean_acc = [mean_acc; mean(accs2)];
end

end
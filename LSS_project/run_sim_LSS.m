function [accs, DATA2] = run_sim_LSS(simulationOptions)

% runs simulations with functions simulateClusteredfMRIData_fullBrain_LSS &
% run_clf_LSS

%Memory footprint can be an issue here...

import rsa.sim.*

%% create covariance matrix

sigma_mat = eye(simulationOptions.signal_voxels); % initialize an identity matrix

R=[];
for cond = 1:simulationOptions.nConditions
    R = [R; mvnrnd(randn(1,simulationOptions.signal_voxels),sigma_mat*simulationOptions.big_sigma)]; % create the mean vectors for the true signal
end

sigma_mat(sigma_mat==0) = simulationOptions.corrs; % add some correlations
cov_mat = wishrnd(sigma_mat*simulationOptions.trial_sigma,round(simulationOptions.cov_mat_df)); % full covariance, depends on df

simulationOptions.R = R; % save the mean vectors
simulationOptions.cov_mat = cov_mat./round(simulationOptions.cov_mat_df); % scale the covariance matrix

R_var = var(simulationOptions.R);
[~,rsv_num] = max(R_var); %best signal voxel (only used if simulations' nf = 1)

for run = 1:simulationOptions.nruns
    %simulate fMRI data
    
    disp('run: ')
    disp(run)
    
    [~,~,~, DATA.(sprintf('run%d',run)).fMRI] = simulateClusteredfMRIData_fullBrain_LSS(simulationOptions);
    
    refX = DATA.(sprintf('run%d',run)).fMRI.X_all.model0; %reference X for all models
    
    true_voxel_msk = DATA.(sprintf('run%d',run)).fMRI.B_true~=0; %non-zero voxel indices for locating signal voxels
    signal_voxel_cols = find(sum(true_voxel_msk));
    
    %probably not the best way to relax signal voxels, need to consider 3D
    %structure
    relaxed_signal_voxel_cols=[]; %relaxing signal voxel columns to take into account smoothing, of interest for classifier feature selection
    for relax = -2:2
        relaxed_signal_voxel_cols = [relaxed_signal_voxel_cols; signal_voxel_cols+relax];
    end
    relaxed_signal_voxel_cols = unique(relaxed_signal_voxel_cols);
    
    num_signal_voxels = numel(find(true_voxel_msk));
    DATA.rand_signal_voxel = signal_voxel_cols(rsv_num); %best signal voxel (only used if simulations' nf = 1)
    
    tmp_B_true = reshape(DATA.(sprintf('run%d',run)).fMRI.B_true(true_voxel_msk),num_signal_voxels,1);
    DATA2.(sprintf('run%d',run)).var0 = var(tmp_B_true);
    
    
    DATA.(sprintf('run%d',run)).fMRI.B_true = []; %clear up more memory
    
    %autocorr(DATA.(sprintf('run%d',run)).fMRI.Y_noisy(:,rsv_num)) %check
    %the autocorrelation function of the data
end

%clear up more memory
if simulationOptions.nf == 1
    %tmp = zeros(size(true_voxel_msk));
    %true_voxel_msk = tmp;
    %true_voxel_msk(:,DATA.rand_signal_voxel) = 1;
    true_voxel_msk = logical(1:simulationOptions.nRepititions);
    num_signal_voxels = simulationOptions.nRepititions;
    tmp_B_true = (1:simulationOptions.nRepititions)';
    for run = 1:simulationOptions.nruns
        %size(DATA.(sprintf('run%d',run)).fMRI.Y_noisy)
        DATA.(sprintf('run%d',run)).fMRI.Y_noisy = DATA.(sprintf('run%d',run)).fMRI.Y_noisy(:,DATA.rand_signal_voxel);
        %size(DATA.(sprintf('run%d',run)).fMRI.Y_noisy)
    end
end
%% run LSS models
i=1;

for run = 1:simulationOptions.nruns
    % saturated model (LSA)
    X = DATA.(sprintf('run%d',run)).fMRI.X_all.model0;
    DATA.(sprintf('run%d',run)).fMRI.B_LSA = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
    tmp_B_LSA = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSA(true_voxel_msk),num_signal_voxels,1);
    [DATA2.(sprintf('run%d',run)).r1,DATA2.(sprintf('run%d',run)).p1] = corr(tmp_B_true,tmp_B_LSA);
    DATA2.(sprintf('run%d',run)).var1 = var(tmp_B_LSA);
    X=[];
end
model = 'B_LSA';
[~, accs(i)] = run_clf_LSS(model,DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
i=i+1;
for run = 1:simulationOptions.nruns
    DATA.(sprintf('run%d',run)).fMRI.(model) = [];
end

if length(simulationOptions.model_list) > 1
    
    % LSS00 %one glm per trial (Mumford et al. method)
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.B_LSS00=[];
        for trial = 1:size(refX,2)
            X = DATA.(sprintf('run%d',run)).fMRI.X_all.model1.(sprintf('num%d',trial));
            B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
            DATA.(sprintf('run%d',run)).fMRI.B_LSS00 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00; B_hats(1,:)];
        end
        tmp_B_LSS00 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS00(true_voxel_msk),num_signal_voxels,1);
        [DATA2.(sprintf('run%d',run)).r2,DATA2.(sprintf('run%d',run)).p2] = corr(tmp_B_true,tmp_B_LSS00);
        DATA2.(sprintf('run%d',run)).var2 = var(tmp_B_LSS00);
        X=[];
    end
    model = 'B_LSS00';
    [~, accs(i)] = run_clf_LSS(model,DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
    i=i+1;
    %DATA.(sprintf('run%d',run)).fMRI.(model) = []; %don't clear this one since
    %it is used for padding
    
    
    % LSS01
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.B_LSS01=[];
        for trial = 1:size(refX,2)-1
            X = DATA.(sprintf('run%d',run)).fMRI.X_all.model2.(sprintf('num%d',trial));
            B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
            DATA.(sprintf('run%d',run)).fMRI.B_LSS01 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS01; B_hats(1,:)];
        end
        DATA.(sprintf('run%d',run)).fMRI.B_LSS01 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS01; DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end,:)]; %do some padding for edges
        tmp_B_LSS01 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS01(true_voxel_msk),num_signal_voxels,1);
        [DATA2.(sprintf('run%d',run)).r3,DATA2.(sprintf('run%d',run)).p3] = corr(tmp_B_true,tmp_B_LSS01);
        DATA2.(sprintf('run%d',run)).var3 = var(tmp_B_LSS01);
        X=[];
    end
    model = 'B_LSS01';
    [~, accs(i)] = run_clf_LSS(model,DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
    i=i+1;
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.(model) = []; %clear variable due to memory overload
    end
    
    
    % LSS10
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.B_LSS10=[];
        for trial = 1:size(refX,2)-1
            X = DATA.(sprintf('run%d',run)).fMRI.X_all.model2.(sprintf('num%d',trial));
            B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
            DATA.(sprintf('run%d',run)).fMRI.B_LSS10 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS10; B_hats(2,:)];
        end
        DATA.(sprintf('run%d',run)).fMRI.B_LSS10 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1,:); DATA.(sprintf('run%d',run)).fMRI.B_LSS10];
        tmp_B_LSS10 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS10(true_voxel_msk),num_signal_voxels,1);
        [DATA2.(sprintf('run%d',run)).r4,DATA2.(sprintf('run%d',run)).p4] = corr(tmp_B_true,tmp_B_LSS10);
        DATA2.(sprintf('run%d',run)).var4 = var(tmp_B_LSS10);
        X=[];
    end
    model = 'B_LSS10';
    [~, accs(i)] = run_clf_LSS(model,DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
    i=i+1;
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.(model) = []; %clear variable due to memory overload
    end
end


if length(simulationOptions.model_list) > 4
    
    % LSS02
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.B_LSS02=[];
        for trial = 1:size(refX,2)-2
            X = DATA.(sprintf('run%d',run)).fMRI.X_all.model3.(sprintf('num%d',trial));
            B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
            DATA.(sprintf('run%d',run)).fMRI.B_LSS02 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS02; B_hats(1,:)];
        end
        DATA.(sprintf('run%d',run)).fMRI.B_LSS02 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS02; DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end-1:end,:)]; %do some padding for edges
        tmp_B_LSS02 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS02(true_voxel_msk),num_signal_voxels,1);
        [DATA2.(sprintf('run%d',run)).r5,DATA2.(sprintf('run%d',run)).p5] = corr(tmp_B_true,tmp_B_LSS02);
        DATA2.(sprintf('run%d',run)).var5 = var(tmp_B_LSS02);
        X=[];
    end
    model = 'B_LSS02';
    [~, accs(i)] = run_clf_LSS(model,DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
    i=i+1;
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.(model) = []; %clear variable due to memory overload   
    end
    
    
    % LSS11
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.B_LSS11=[];
        for trial = 1:size(refX,2)-2
            X = DATA.(sprintf('run%d',run)).fMRI.X_all.model3.(sprintf('num%d',trial));
            B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
            DATA.(sprintf('run%d',run)).fMRI.B_LSS11 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS11; B_hats(2,:)];
        end
        DATA.(sprintf('run%d',run)).fMRI.B_LSS11 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1,:); DATA.(sprintf('run%d',run)).fMRI.B_LSS11; DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end,:)];
        tmp_B_LSS11 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS11(true_voxel_msk),num_signal_voxels,1);
        [DATA2.(sprintf('run%d',run)).r6,DATA2.(sprintf('run%d',run)).p6] = corr(tmp_B_true,tmp_B_LSS11);
        DATA2.(sprintf('run%d',run)).var6 = var(tmp_B_LSS11);
        X=[];
    end    
    model = 'B_LSS11';
    [~, accs(i)] = run_clf_LSS(model,DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
    i=i+1;
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.(model) = []; %clear variable due to memory overload    
    end
    
    
    % LSS20
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.B_LSS20=[];
        for trial = 1:size(refX,2)-2
            X = DATA.(sprintf('run%d',run)).fMRI.X_all.model3.(sprintf('num%d',trial));
            B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
            DATA.(sprintf('run%d',run)).fMRI.B_LSS20 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS20; B_hats(3,:)];
        end
        DATA.(sprintf('run%d',run)).fMRI.B_LSS20 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1:2,:); DATA.(sprintf('run%d',run)).fMRI.B_LSS20];
        tmp_B_LSS20 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS20(true_voxel_msk),num_signal_voxels,1);
        [DATA2.(sprintf('run%d',run)).r7,DATA2.(sprintf('run%d',run)).p7] = corr(tmp_B_true,tmp_B_LSS20);
        DATA2.(sprintf('run%d',run)).var7 = var(tmp_B_LSS20);
        X=[];
    end    
    model = 'B_LSS20';
    [~, accs(i)] = run_clf_LSS(model,DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
    i=i+1;
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.(model) = []; %clear variable due to memory overload    
    end
    
    
    % LSS12
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.B_LSS12=[];
        for trial = 1:size(refX,2)-3
            X = DATA.(sprintf('run%d',run)).fMRI.X_all.model4.(sprintf('num%d',trial));
            B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
            DATA.(sprintf('run%d',run)).fMRI.B_LSS12 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS12; B_hats(2,:)];
        end
        DATA.(sprintf('run%d',run)).fMRI.B_LSS12 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1,:); ...
            DATA.(sprintf('run%d',run)).fMRI.B_LSS12; DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end-1:end,:)]; %do some padding for edges
        tmp_B_LSS12 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS12(true_voxel_msk),num_signal_voxels,1);
        [DATA2.(sprintf('run%d',run)).r8,DATA2.(sprintf('run%d',run)).p8] = corr(tmp_B_true,tmp_B_LSS12);
        DATA2.(sprintf('run%d',run)).var8 = var(tmp_B_LSS12);
        X=[];
    end
    model = 'B_LSS12';
    [~, accs(i)] = run_clf_LSS(model,DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
    i=i+1;
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.(model) = []; %clear variable due to memory overload
    end
    
    
    % LSS21
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.B_LSS21=[];
        for trial = 1:size(refX,2)-3
            X = DATA.(sprintf('run%d',run)).fMRI.X_all.model4.(sprintf('num%d',trial));
            B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
            DATA.(sprintf('run%d',run)).fMRI.B_LSS21 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS21; B_hats(3,:)];
        end
        DATA.(sprintf('run%d',run)).fMRI.B_LSS21 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1:2,:); DATA.(sprintf('run%d',run)).fMRI.B_LSS21; ...
            DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end,:)];
        tmp_B_LSS21 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS21(true_voxel_msk),num_signal_voxels,1);
        [DATA2.(sprintf('run%d',run)).r9,DATA2.(sprintf('run%d',run)).p9] = corr(tmp_B_true,tmp_B_LSS21);
        DATA2.(sprintf('run%d',run)).var9 = var(tmp_B_LSS21);
        X=[];
    end    
    model = 'B_LSS21';
    [~, accs(i)] = run_clf_LSS(model,DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
    i=i+1;
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.(model) = []; %clear variable due to memory overload    
    end
    
    
    % LSS22
    model = 'B_LSS22'; %simulationOptions.model_list{i}
    model_num = '5';
    target_trial = 3;
    %DATA, refX
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.(model)=[];
        for trial = 1:size(refX,2)-(str2num(model_num)-1)
            X = DATA.(sprintf('run%d',run)).fMRI.X_all.(['model' model_num]).(sprintf('num%d',trial));
            B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
            DATA.(sprintf('run%d',run)).fMRI.(model) = [DATA.(sprintf('run%d',run)).fMRI.(model); B_hats(target_trial,:)];
        end
        DATA.(sprintf('run%d',run)).fMRI.(model) = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1:2,:); DATA.(sprintf('run%d',run)).fMRI.(model); ...
            DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end-1:end,:)];
        tmp_B_LSS22 = reshape(DATA.(sprintf('run%d',run)).fMRI.(model)(true_voxel_msk),num_signal_voxels,1);
        [DATA2.(sprintf('run%d',run)).r10,DATA2.(sprintf('run%d',run)).p10] = corr(tmp_B_true,tmp_B_LSS22);
        DATA2.(sprintf('run%d',run)).var10 = var(tmp_B_LSS22);
        X=[];
    end
    [~, accs(i)] = run_clf_LSS(model,DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
    %i=i+1;
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.(model) = []; %clear variable due to memory overload
    end
    
end

for run = 1:simulationOptions.nruns
    DATA.(sprintf('run%d',run)).fMRI.B_LSS00=[]; %since padding finished on all models, clear up more memory
end

%DATA2.(sprintf('run%d',run)).X = DATA.(sprintf('run%d',run)).fMRI.X;
%DATA2.(sprintf('run%d',run)).X_all = DATA.(sprintf('run%d',run)).fMRI.X_all;
DATA2.(sprintf('run%d',run)).groundTruth = DATA.(sprintf('run%d',run)).fMRI.groundTruth;
DATA2.(sprintf('run%d',run)).sequence = DATA.(sprintf('run%d',run)).fMRI.sequence;
%DATA2.(sprintf('run%d',run)).b = DATA.(sprintf('run%d',run)).fMRI.b;
DATA2.(sprintf('run%d',run)).volumeSize_vox = DATA.(sprintf('run%d',run)).fMRI.volumeSize_vox;
DATA2.(sprintf('run%d',run)).simulationOptions = simulationOptions;

%% run classifiers (used to be run all at once but took up too much memory)
%for i = 1:length(simulationOptions.model_list)
%     model = 'LSA'; %simulationOptions.model_list{i}
%     [~, accs(i)] = run_clf_LSS(model,DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
%     i=i+1;
%     if simulationOptions.nf > 1
%         %compute overlap between feature selection & real signal voxels
%         feat_sel_ratios=[];
%         feat_sel_ratios_relaxed=[];
%         for j = 1:simulationOptions.nruns
%             feat_sel_ratios(j) = numel(intersect(signal_voxel_cols,chosen_voxels(j,:)))./numel(chosen_voxels(j,:)); %ratio of chosen voxels that were true signal, given number of features (nf) selected for classifier
%             feat_sel_ratios_relaxed(j) = numel(intersect(relaxed_signal_voxel_cols,chosen_voxels(j,:)))./numel(chosen_voxels(j,:)); %relaxing signal voxels due to smoothing
%         end
%
%         all_feat_sel_ratios(i) = mean(feat_sel_ratios);
%         all_feat_sel_ratios_relaxed(i) = mean(feat_sel_ratios_relaxed);
%     end
%end
% if simulationOptions.nf > 1
%     DATA2.(sprintf('run%d',run)).all_feat_sel_ratios = all_feat_sel_ratios;
%     DATA2.(sprintf('run%d',run)).all_feat_sel_ratios_relaxed = all_feat_sel_ratios_relaxed;
% else
%     DATA2.(sprintf('run%d',run)).all_feat_sel_ratios = [];
%     DATA2.(sprintf('run%d',run)).all_feat_sel_ratios_relaxed = [];
% end

end


function [accs, DATA2] = run_sim_LSS(simulationOptions)

% runs simulations with functions simulateClusteredfMRIData_fullBrain_LSS &
% run_clf_LSS

%Memory footprint can be an issue here, especially if running on multiple
%cores. Probably best to run the classifier after estimating each model and
%then clearing the model from memory (by nesting the for loops).

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
    
    refX = DATA.(sprintf('run%d',run)).fMRI.X; %reference X for all models
    
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
    
    %autocorr(DATA.(sprintf('run%d',run)).fMRI.Y_noisy(:,rsv_num)) %check
    %the autocorrelation function of the data
    %% run LSS models
    
    % saturated model (LSA)
    X = refX;
    DATA.(sprintf('run%d',run)).fMRI.B_LSA = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
    tmp_B_LSA = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSA(true_voxel_msk),num_signal_voxels,1);
    [DATA2.(sprintf('run%d',run)).r1,DATA2.(sprintf('run%d',run)).p1] = corr(tmp_B_true,tmp_B_LSA);
    DATA2.(sprintf('run%d',run)).var1 = var(tmp_B_LSA);
    X=[];
    
    % LSS0_0 %one glm per trial (Mumford et al. method)
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
    
    % LSS01 & LSS10
    DATA.(sprintf('run%d',run)).fMRI.B_LSS01=[]; DATA.(sprintf('run%d',run)).fMRI.B_LSS10=[];
    for trial = 1:size(refX,2)-1
        X = DATA.(sprintf('run%d',run)).fMRI.X_all.model2.(sprintf('num%d',trial));
        B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
        DATA.(sprintf('run%d',run)).fMRI.B_LSS01 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS01; B_hats(1,:)];
        DATA.(sprintf('run%d',run)).fMRI.B_LSS10 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS10; B_hats(2,:)];
    end
    DATA.(sprintf('run%d',run)).fMRI.B_LSS01 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS01; DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end,:)]; %do some padding for edges
    DATA.(sprintf('run%d',run)).fMRI.B_LSS10 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1,:); DATA.(sprintf('run%d',run)).fMRI.B_LSS10];
    tmp_B_LSS01 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS01(true_voxel_msk),num_signal_voxels,1);
    [DATA2.(sprintf('run%d',run)).r3,DATA2.(sprintf('run%d',run)).p3] = corr(tmp_B_true,tmp_B_LSS01);
    DATA2.(sprintf('run%d',run)).var3 = var(tmp_B_LSS01);
    
    tmp_B_LSS10 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS10(true_voxel_msk),num_signal_voxels,1);
    [DATA2.(sprintf('run%d',run)).r4,DATA2.(sprintf('run%d',run)).p4] = corr(tmp_B_true,tmp_B_LSS10);
    DATA2.(sprintf('run%d',run)).var4 = var(tmp_B_LSS10);
    X=[];
    
    if length(simulationOptions.model_list) > 4
    
    % LSS02 & LSS11 & LSS20
    DATA.(sprintf('run%d',run)).fMRI.B_LSS02=[]; DATA.(sprintf('run%d',run)).fMRI.B_LSS11=[]; DATA.(sprintf('run%d',run)).fMRI.B_LSS20=[];
    for trial = 1:size(refX,2)-2
        X = DATA.(sprintf('run%d',run)).fMRI.X_all.model3.(sprintf('num%d',trial));
        B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
        DATA.(sprintf('run%d',run)).fMRI.B_LSS02 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS02; B_hats(1,:)];
        DATA.(sprintf('run%d',run)).fMRI.B_LSS11 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS11; B_hats(2,:)];
        DATA.(sprintf('run%d',run)).fMRI.B_LSS20 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS20; B_hats(3,:)];
    end
    DATA.(sprintf('run%d',run)).fMRI.B_LSS02 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS02; DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end-1:end,:)]; %do some padding for edges
    DATA.(sprintf('run%d',run)).fMRI.B_LSS11 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1,:); DATA.(sprintf('run%d',run)).fMRI.B_LSS11; DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end,:)];
    DATA.(sprintf('run%d',run)).fMRI.B_LSS20 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1:2,:); DATA.(sprintf('run%d',run)).fMRI.B_LSS20];
    tmp_B_LSS02 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS02(true_voxel_msk),num_signal_voxels,1);
    [DATA2.(sprintf('run%d',run)).r5,DATA2.(sprintf('run%d',run)).p5] = corr(tmp_B_true,tmp_B_LSS02);
    DATA2.(sprintf('run%d',run)).var5 = var(tmp_B_LSS02);
    
    tmp_B_LSS11 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS11(true_voxel_msk),num_signal_voxels,1);
    [DATA2.(sprintf('run%d',run)).r6,DATA2.(sprintf('run%d',run)).p6] = corr(tmp_B_true,tmp_B_LSS11);
    DATA2.(sprintf('run%d',run)).var6 = var(tmp_B_LSS11);
    
    tmp_B_LSS20 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS20(true_voxel_msk),num_signal_voxels,1);
    [DATA2.(sprintf('run%d',run)).r7,DATA2.(sprintf('run%d',run)).p7] = corr(tmp_B_true,tmp_B_LSS20);
    DATA2.(sprintf('run%d',run)).var7 = var(tmp_B_LSS20);
    X=[];
    
    % LSS12 & LSS21
    DATA.(sprintf('run%d',run)).fMRI.B_LSS12=[]; DATA.(sprintf('run%d',run)).fMRI.B_LSS21=[];
    for trial = 1:size(refX,2)-3
        X = DATA.(sprintf('run%d',run)).fMRI.X_all.model4.(sprintf('num%d',trial));
        B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;     
        DATA.(sprintf('run%d',run)).fMRI.B_LSS12 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS12; B_hats(2,:)];
        DATA.(sprintf('run%d',run)).fMRI.B_LSS21 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS21; B_hats(3,:)];        
    end
    DATA.(sprintf('run%d',run)).fMRI.B_LSS12 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1,:); ...
        DATA.(sprintf('run%d',run)).fMRI.B_LSS12; DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end-1:end,:)]; %do some padding for edges
    DATA.(sprintf('run%d',run)).fMRI.B_LSS21 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1:2,:); DATA.(sprintf('run%d',run)).fMRI.B_LSS21; ...
        DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end,:)];
    tmp_B_LSS12 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS12(true_voxel_msk),num_signal_voxels,1);
    [DATA2.(sprintf('run%d',run)).r8,DATA2.(sprintf('run%d',run)).p8] = corr(tmp_B_true,tmp_B_LSS12);
    DATA2.(sprintf('run%d',run)).var8 = var(tmp_B_LSS12);
    
    tmp_B_LSS21 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS21(true_voxel_msk),num_signal_voxels,1);
    [DATA2.(sprintf('run%d',run)).r9,DATA2.(sprintf('run%d',run)).p9] = corr(tmp_B_true,tmp_B_LSS21);
    DATA2.(sprintf('run%d',run)).var9 = var(tmp_B_LSS21);
    X=[];
    
    % LSS22
    DATA.(sprintf('run%d',run)).fMRI.B_LSS22=[];
    for trial = 1:size(refX,2)-4
        X = DATA.(sprintf('run%d',run)).fMRI.X_all.model5.(sprintf('num%d',trial));
        B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;     
        DATA.(sprintf('run%d',run)).fMRI.B_LSS22 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS22; B_hats(3,:)];        
    end
    DATA.(sprintf('run%d',run)).fMRI.B_LSS22 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00(1:2,:); DATA.(sprintf('run%d',run)).fMRI.B_LSS22; ...
        DATA.(sprintf('run%d',run)).fMRI.B_LSS00(end-1:end,:)];
    tmp_B_LSS22 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS22(true_voxel_msk),num_signal_voxels,1);
    [DATA2.(sprintf('run%d',run)).r10,DATA2.(sprintf('run%d',run)).p10] = corr(tmp_B_true,tmp_B_LSS22);
    DATA2.(sprintf('run%d',run)).var10 = var(tmp_B_LSS22);
    X=[];
    
    end
    

DATA2.(sprintf('run%d',run)).X = DATA.(sprintf('run%d',run)).fMRI.X;
DATA2.(sprintf('run%d',run)).X_all = DATA.(sprintf('run%d',run)).fMRI.X_all;
DATA2.(sprintf('run%d',run)).groundTruth = DATA.(sprintf('run%d',run)).fMRI.groundTruth;
DATA2.(sprintf('run%d',run)).sequence = DATA.(sprintf('run%d',run)).fMRI.sequence;
%DATA2.(sprintf('run%d',run)).b = DATA.(sprintf('run%d',run)).fMRI.b;
DATA2.(sprintf('run%d',run)).volumeSize_vox = DATA.(sprintf('run%d',run)).fMRI.volumeSize_vox;
DATA2.(sprintf('run%d',run)).simulationOptions = simulationOptions;

%transfer estimated parameters for saving... not plausible since these are
%whole brain, if needed save direct to disk, don't hold in memory
%DATA2.(sprintf('run%d',run)).B_LSA = DATA.(sprintf('run%d',run)).fMRI.B_LSA;
%DATA2.(sprintf('run%d',run)).B_LSS00 = DATA.(sprintf('run%d',run)).fMRI.B_LSS00;
%DATA2.(sprintf('run%d',run)).B_LSS01 = DATA.(sprintf('run%d',run)).fMRI.B_LSS01;
%DATA2.(sprintf('run%d',run)).B_LSS10 = DATA.(sprintf('run%d',run)).fMRI.B_LSS10;
%DATA2.(sprintf('run%d',run)).B_LSS02 = DATA.(sprintf('run%d',run)).fMRI.B_LSS02;
%DATA2.(sprintf('run%d',run)).B_LSS11 = DATA.(sprintf('run%d',run)).fMRI.B_LSS11;
%DATA2.(sprintf('run%d',run)).B_LSS20 = DATA.(sprintf('run%d',run)).fMRI.B_LSS20;
%DATA2.(sprintf('run%d',run)).B_LSS12 = DATA.(sprintf('run%d',run)).fMRI.B_LSS12;
%DATA2.(sprintf('run%d',run)).B_LSS21 = DATA.(sprintf('run%d',run)).fMRI.B_LSS21;
%DATA2.(sprintf('run%d',run)).B_LSS22 = DATA.(sprintf('run%d',run)).fMRI.B_LSS22;
end

%% run classifiers

for i = 1:length(simulationOptions.model_list)
    [chosen_voxels, accs(i)] = run_clf_LSS(simulationOptions.model_list{i},DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
    
    if simulationOptions.nf > 1
        %compute overlap between feature selection & real signal voxels
        feat_sel_ratios=[];
        feat_sel_ratios_relaxed=[];
        for j = 1:simulationOptions.nruns
            feat_sel_ratios(j) = numel(intersect(signal_voxel_cols,chosen_voxels(j,:)))./numel(chosen_voxels(j,:)); %ratio of chosen voxels that were true signal, given number of features (nf) selected for classifier
            feat_sel_ratios_relaxed(j) = numel(intersect(relaxed_signal_voxel_cols,chosen_voxels(j,:)))./numel(chosen_voxels(j,:)); %relaxing signal voxels due to smoothing
        end
        
        all_feat_sel_ratios(i) = mean(feat_sel_ratios);
        all_feat_sel_ratios_relaxed(i) = mean(feat_sel_ratios_relaxed);
    end
    
end

if simulationOptions.nf > 1
    DATA2.(sprintf('run%d',run)).all_feat_sel_ratios = all_feat_sel_ratios;
    DATA2.(sprintf('run%d',run)).all_feat_sel_ratios_relaxed = all_feat_sel_ratios_relaxed;
else
    DATA2.(sprintf('run%d',run)).all_feat_sel_ratios = [];
    DATA2.(sprintf('run%d',run)).all_feat_sel_ratios_relaxed = [];    
end

end


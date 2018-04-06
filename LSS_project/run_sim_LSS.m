function [accs, DATA2] = run_sim_LSS(simulationOptions)

% runs simulations with functions simulateClusteredfMRIData_fullBrain_LSS &
% run_clf_LSS

%Memory footprint can be an issue here, especially if running on multiple
%cores. Probably best to run the classifier after estimating each model and
%then clearing the model from memory.

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


for run = 1:simulationOptions.nruns
    %simulate fMRI data
    
    disp('run: ')
    disp(run)
    
    [~,~,~, DATA.(sprintf('run%d',run)).fMRI] = simulateClusteredfMRIData_fullBrain_LSS(simulationOptions);
    
    refX = DATA.(sprintf('run%d',run)).fMRI.X; %reference X for all models
    
    %% run LSS models
    
    % saturated model (LSA)
    X = refX;
    DATA.(sprintf('run%d',run)).fMRI.B_LSA = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
    brain_size = numel(DATA.(sprintf('run%d',run)).fMRI.B_LSA);
    [DATA2.(sprintf('run%d',run)).r1,DATA2.(sprintf('run%d',run)).p1] = corr(reshape(DATA.(sprintf('run%d',run)).fMRI.B_true,brain_size,1),reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSA,brain_size,1));
    X=[];
    
    % LSS0_0 %one glm per trial (Mumford et al. method)
    DATA.(sprintf('run%d',run)).fMRI.B_LSS00=[];
    for trial = 1:size(refX,2)
        X = DATA.(sprintf('run%d',run)).fMRI.X_all.model1.(sprintf('num%d',trial));
        B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
        DATA.(sprintf('run%d',run)).fMRI.B_LSS00 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00; B_hats(1,:)];
    end
    [DATA2.(sprintf('run%d',run)).r2,DATA2.(sprintf('run%d',run)).p2] = corr(reshape(DATA.(sprintf('run%d',run)).fMRI.B_true,brain_size,1),reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS00,brain_size,1));
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
    [DATA2.(sprintf('run%d',run)).r3,DATA2.(sprintf('run%d',run)).p3] = corr(reshape(DATA.(sprintf('run%d',run)).fMRI.B_true,brain_size,1),reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS01,brain_size,1));
    [DATA2.(sprintf('run%d',run)).r4,DATA2.(sprintf('run%d',run)).p4] = corr(reshape(DATA.(sprintf('run%d',run)).fMRI.B_true,brain_size,1),reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS10,brain_size,1));
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
    [DATA2.(sprintf('run%d',run)).r5,DATA2.(sprintf('run%d',run)).p5] = corr(reshape(DATA.(sprintf('run%d',run)).fMRI.B_true,brain_size,1),reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS02,brain_size,1));
    [DATA2.(sprintf('run%d',run)).r6,DATA2.(sprintf('run%d',run)).p6] = corr(reshape(DATA.(sprintf('run%d',run)).fMRI.B_true,brain_size,1),reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS11,brain_size,1));
    [DATA2.(sprintf('run%d',run)).r7,DATA2.(sprintf('run%d',run)).p7] = corr(reshape(DATA.(sprintf('run%d',run)).fMRI.B_true,brain_size,1),reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS20,brain_size,1));
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
    [DATA2.(sprintf('run%d',run)).r8,DATA2.(sprintf('run%d',run)).p8] = corr(reshape(DATA.(sprintf('run%d',run)).fMRI.B_true,brain_size,1),reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS12,brain_size,1));
    [DATA2.(sprintf('run%d',run)).r9,DATA2.(sprintf('run%d',run)).p9] = corr(reshape(DATA.(sprintf('run%d',run)).fMRI.B_true,brain_size,1),reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS21,brain_size,1));
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
    [DATA2.(sprintf('run%d',run)).r10,DATA2.(sprintf('run%d',run)).p10] = corr(reshape(DATA.(sprintf('run%d',run)).fMRI.B_true,brain_size,1),reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS22,brain_size,1));
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
    accs(i) = run_clf_LSS(simulationOptions.model_list{i},DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
end

end


function [accs, DATA2] = run_sim_LSS(simulationOptions)

% runs simulations with functions simulateClusteredfMRIData_fullBrain_LSS &
% run_clf_LSS

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
    X=[];
    
    % LSS0_0 %one glm per trial (Mumford et al. method)
    DATA.(sprintf('run%d',run)).fMRI.B_LSS00=[];
    for trial = 1:size(refX,2)
        X = DATA.(sprintf('run%d',run)).fMRI.X_all.model1.(sprintf('num%d',trial));
        B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
        DATA.(sprintf('run%d',run)).fMRI.B_LSS00 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00; B_hats(1,:)];
    end
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
    X=[];
    
    end
    

DATA2.(sprintf('run%d',run)).X = DATA.(sprintf('run%d',run)).fMRI.X;
DATA2.(sprintf('run%d',run)).groundTruth = DATA.(sprintf('run%d',run)).fMRI.groundTruth;
DATA2.(sprintf('run%d',run)).sequence = DATA.(sprintf('run%d',run)).fMRI.sequence;
DATA2.(sprintf('run%d',run)).b = DATA.(sprintf('run%d',run)).fMRI.b;
DATA2.(sprintf('run%d',run)).volumeSize_vox = DATA.(sprintf('run%d',run)).fMRI.volumeSize_vox;
DATA2.(sprintf('run%d',run)).simulationOptions = simulationOptions;
end

%% run classifiers

for i = 1:length(simulationOptions.model_list)
    accs(i) = run_clf_LSS(simulationOptions.model_list{i},DATA,simulationOptions.nConditions,simulationOptions.nruns,simulationOptions.nf,simulationOptions.svm_options);
end

end


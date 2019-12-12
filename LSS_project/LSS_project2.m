clear all; close all; clc;

%dependencies: rsatoolbox, libsvm,
%run_clf_LSS, run_sim_LSS,
%simulateClusteredfMRIData_fullBrain_LSS (modified from rsatoolbox),
%generateCognitiveModel_fastButTrialsNeedToStartOnVols_LSS (modified from rsatoolbox)

%simulationOptions structure below was taken from the rsatoolbox script
%called simulationOptions_demo

% Script that models BOLD signal with covariance amongst voxels and
% compares LSS models (cf. Mumford et al., 2012)
% by Sebastian Bobadilla Suarez

%setup paths before running
toolboxRoot = '/media/seb/HD_Numba_Juan/Dropbox/postdoc/rsatoolbox/';  %parent directory for rsatoolbox
%toolboxRoot = '/mnt/love12/rsatoolbox/';
cd(toolboxRoot)
addpath(genpath(toolboxRoot)); %add all sub-paths to path

%% setup model parameters

parjob = 0; %1 if using cluster, runs njobs
njobs = 500; %number of jobs per noise setting, only if running on many cores

big_brain = 0; %if big_brain==1 then size is [64 64 32], otherwise the triple of simulationOptions.volumeSize_vox

% A 4-tuple. The first three entries are the x, y and z values for the gaussian
% spatial smoothing kernel FWHM in mm and the fourth is the size of the temporal
% smoothing FWHM.
smoothing_kernel = [4 4 4 4.5] ;
tmp = num2str(smoothing_kernel);
skernel_str = tmp(find(~isspace(tmp)));
simulationOptions.spatiotemporalSmoothingFWHM_mm_s = smoothing_kernel;

simulationOptions.lambda_list = [linspace(0,10,10) linspace(100,10000,10)];

% The amount of noise to be added by the simulated scanner. This corresponds to
% the square of the standard deviation of the gaussian distibution from which
% the noise is drawn (?).
simulationOptions.scannerNoiseLevel = 10000; %10000./1000; % used to be 3000

simulationOptions.nRepititions = 20; %20 %500; %repetitions per stimulus

simulationOptions.nruns = 2; %3 %separate fMRI blocks
simulationOptions.ridge = 2; %needs 2 runs, 1=optimize, 2=use lambda list
simulationOptions.stimulusDuration = 1.5; %in seconds

simulationOptions.TR = 1; %scanner TR
simulationOptions.nConditions = 2; %2 %how many stimulus conditions (i.e., classes)?

%setup classifier information
%if logistic regression is preferred to svm then: simulationOptions.svm_options = 0;
%otherwise, pass in a string with libsvm parameters
%simulationOptions.svm_options = 0;
%simulationOptions.svm_options = '-s 0 -t 0 -n 0.5 -h 0'; % -t 0 is linear kernel, -t 2 rbf, -h 0 is faster (shrinking heuristic)
% check libsvm website for more information ondelete(gcp('nocreate')) options
simulationOptions.nf = 300; %1; %300 %number of features (voxels) that go into the classifier (used 20 for logistic regression & 300 for linear SVM)
%if nf=1 then a random signal voxel is chosen

all_models = 1;
if all_models==0
    simulationOptions.model_list = {'B_LSA','B_LSS00','B_LSS01','B_LSS10'};
elseif simulationOptions.ridge ==0
    simulationOptions.model_list = {'B_LSA','B_LSS00','B_LSS01','B_LSS10','B_LSS02','B_LSS11','B_LSS20','B_LSS12','B_LSS21','B_LSS22'};
elseif simulationOptions.ridge == 1
    simulationOptions.model_list = {'B_LSA','B_LSS00','B_LSS01','B_LSS10','B_LSS02','B_LSS11','B_LSS20','B_LSS12','B_LSS21','B_LSS22','B_ridge'};
elseif simulationOptions.ridge == 2
    simulationOptions.model_list = {'B_LSA','B_LSS00','B_LSS01','B_LSS10','B_LSS02','B_LSS11','B_LSS20','B_LSS12','B_LSS21','B_LSS22'};
    for i = 1:length(simulationOptions.lambda_list)
        if simulationOptions.lambda_list(i)<10
            simulationOptions.model_list{10+i} =  ['lam(a)' num2str(simulationOptions.lambda_list(i))];
        else
            simulationOptions.model_list{10+i} =  ['lam(b)' num2str(simulationOptions.lambda_list(i))];
        end
    end
end

simulationOptions.trial_sigma = 1; %0.5^2; %variance between trials for each stimulus condition
simulationOptions.volumeSize_vox = [7 7 7]; % size of the signal
simulationOptions.signal_voxels = prod(simulationOptions.volumeSize_vox); % number of signal voxels
simulationOptions.exp = 1; %controls variance of the wishart distribution
simulationOptions.cov_mat_df = simulationOptions.signal_voxels^simulationOptions.exp; % controls variance of the wishart distribution
simulationOptions.corrs = 0.7; %^2; %correlations for the covariance matrix from which trial vectors for each stimulus are sampled

% A triple containing the dimensions of one voxel in mm.
simulationOptions.voxelSize_mm = [3 3 3.75];

%%sp_smoothing = 4441; %controls spatial smoothing (0 for none), if -1 then also kills temporal smoothing, if >2 then temporal smoothing == sp_smoothing
%4441 kills temporal but keeps spatial smoothing
% if sp_smoothing==0
%     simulationOptions.spatiotemporalSmoothingFWHM_mm_s = [1 1 1 4.5]; %% %
% elseif sp_smoothing==1
%     simulationOptions.spatiotemporalSmoothingFWHM_mm_s = [4 4 4 4.5];
% elseif sp_smoothing==-1
%     simulationOptions.spatiotemporalSmoothingFWHM_mm_s = [1 1 1 1]; %% %
% elseif sp_smoothing<2
%     simulationOptions.spatiotemporalSmoothingFWHM_mm_s = [1 1 1 sp_smoothing];
% elseif sp_smoothing==4441
%     simulationOptions.spatiotemporalSmoothingFWHM_mm_s = [4 4 4 1];
% end


%% setup the noise levels and collinearity (through trial duration)

% with these two lists, 8 levels will be run in totaldelete(gcp('nocreate'))

trial_duration_list = [3 4]; % in secondsdelete(gcp('nocreate'))
%big_sigma_list = [10 10 15 15 20 20];%linspace(10, 10.1,simulationOptions.nConditions); %[15 10]; %[10, 15, 20]; % sigma for the hyper-ellipse containing the mean random vectors
%scan_noise_list = [3000, 5000, 7000, 10000]; %if needed you can add a for
%loop below with different levels of scanner noise

if big_brain==1
    simulationOptions.brainVol = [64 64 32]; %%
    simulationOptions.effectCen = [20 20 15];
    %where to save the simulation results
    save_path = ['/media/seb/HD_Numba_Juan/ridge_results_' num2str(simulationOptions.nf) 'voxel_smoothing' skernel_str '_noise' num2str(simulationOptions.scannerNoiseLevel) '_' num2str(simulationOptions.nRepititions) 'samps_bigBrain'];
elseif big_brain==0
    simulationOptions.brainVol = simulationOptions.volumeSize_vox*3;
    simulationOptions.effectCen = [randi(round(simulationOptions.volumeSize_vox(1)*1.5)) randi(round(simulationOptions.volumeSize_vox(2)*1.5)) randi(round(simulationOptions.volumeSize_vox(3)*1.5))];
    %where to save the simulation results
    save_path = ['/media/seb/HD_Numba_Juan/ridge_results_' num2str(simulationOptions.nf) 'voxel_smoothing' skernel_str '_noise' num2str(simulationOptions.scannerNoiseLevel) '_' num2str(simulationOptions.nRepititions) 'samps'];
    %save_path = ['/media/seb/HD_Numba_Juan/ridge_results_' num2str(trial_duration_list(1)) 's_' skernel_str '_noise' num2str(simulationOptions.scannerNoiseLevel) '_' num2str(simulationOptions.nRepititions) 'samps'];
end

disp(save_path)



%% run simulations

for trial_duration = 1:length(trial_duration_list)
    big_sigma_list = [10 10]; %each iteration adds 5 to the noise
    for big_sigma = 1:3 %(length(big_sigma_list))/2 %for scan_noise = 1:length(scan_noise_list) %

        if parjob==0
            simulationOptions.trialDuration = trial_duration_list(trial_duration);
            simulationOptions.big_sigma = big_sigma_list;
            simulationOptions.scannerNoiseLevel = simulationOptions.scannerNoiseLevel;
            [metric, DATA] = run_sim_LSS2(simulationOptions);
            disp(simulationOptions.model_list)
            disp(mean(metric,1))
        else
            simulationOptions.trialDuration = trial_duration_list(trial_duration);
            simulationOptions.big_sigma = big_sigma_list; %(big_sigma);
            %simulationOptions.scannerNoiseLevel = scan_noise_list(scan_noise);

            nworkers = 30;
            SIMS=[];
            all_metrics=[];
            %all_sigma_est=[]; %only coded for the 1 voxel setting
            parpool(nworkers)

            parfor i=1:njobs
                %[sigma_est, DATA] = run_sim_LSS(simulationOptions);
                [metric, DATA] = run_sim_LSS2(simulationOptions);
                all_metrics = [all_metrics; metric];
                %all_sigma_est = [all_sigma_est; sigma_est];
                SIMS = [SIMS, DATA];
            end

            delete(gcp('nocreate'))
            disp(simulationOptions.model_list)
            disp(mean(all_metrics,1))

            if exist(save_path,'dir')==0
                mkdir(save_path)
            end

            cd(save_path)
            %save(['bsig',num2str(simulationOptions.big_sigma),'tsig',num2str(simulationOptions.trial_sigma),'isi', num2str(simulationOptions.trialDuration),'.mat'],'SIMS','all_sigma_est','-v7.3')
            save(['bsig',num2str(simulationOptions.big_sigma(1)),'tsig',num2str(simulationOptions.trial_sigma),'isi', num2str(simulationOptions.trialDuration),'.mat'],'SIMS','all_metrics','simulationOptions','-v7.3')
            %save(['scan_sig',num2str(simulationOptions.scannerNoiseLevel),'tsig',num2str(simulationOptions.trial_sigma),'isi', num2str(simulationOptions.trialDuration),'.mat'],'SIMS')

            cd([toolboxRoot '/LSS_project/'])
        end

        big_sigma_list = big_sigma_list + 5;
    end
end

if parjob==0
bar(categorical(simulationOptions.model_list),sqrt(metric(1,:)))
%pause
end

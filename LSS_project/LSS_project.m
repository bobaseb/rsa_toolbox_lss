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
toolboxRoot = '/home/seb/Documents/rsatoolbox/';  %parent directory for rsatoolbox
cd(toolboxRoot)
addpath(genpath(toolboxRoot)); %add all sub-paths to path

% addpath to the libsvm toolbox in case it's not in your matlab path
addpath('/home/seb/Documents/libsvm-3.22/matlab');

%% setup model parameters

parjob = 1; %1 if using cluster, runs njobs
njobs = 50; %number of jobs per noise setting, only if running on many cores

big_brain = 0; %if big_brain==1 then size is [64 64 32], otherwise the triple of simulationOptions.volumeSize_vox

sp_smoothing = 0; %controls spatial smoothing

% The amount of noise to be added by the simulated scanner. This corresponds to
% the square of the standard deviation of the gaussian distibution from which
% the noise is drawn (?).
simulationOptions.scannerNoiseLevel = 10000; %3000; %10000./1000; % used to be 3000

simulationOptions.nRepititions = 20; %500; %repetitions per stimulus

simulationOptions.nruns = 3; %separate fMRI blocks
simulationOptions.stimulusDuration = 1.5; %in seconds

simulationOptions.TR = 1; %scanner TR
simulationOptions.nConditions = 2; %how many stimulus conditions (i.e., classes)?

%setup classifier information
%if logistic regression is preferred to svm then: simulationOptions.svm_options = 0;
%otherwise, pass in a string with libsvm parameters
%simulationOptions.svm_options = 0;
simulationOptions.svm_options = '-s 0 -t 0 -n 0.5 -h 0'; % -t 0 is linear kernel, -t 2 rbf, -h 0 is faster (shrinking heuristic)
% check libsvm website for more information on options
simulationOptions.nf = 300; %1; %number of features (voxels) that go into the classifier (used 20 for logistic regression & 300 for linear SVM)
%if nf=1 then a random signal voxel is chosen

all_models = 1;
if all_models==0
    simulationOptions.model_list = {'B_LSA','B_LSS00','B_LSS01','B_LSS10'};
else
    simulationOptions.model_list = {'B_LSA','B_LSS00','B_LSS01','B_LSS10','B_LSS02','B_LSS11','B_LSS20','B_LSS12','B_LSS21','B_LSS22'};
end

simulationOptions.trial_sigma = 0.5^2; %variance between trials for each stimulus condition
simulationOptions.volumeSize_vox = [7 7 7]; % size of the signal
simulationOptions.signal_voxels = prod(simulationOptions.volumeSize_vox); % number of signal voxels
simulationOptions.exp = 1; %controls variance of the wishart distribution
simulationOptions.cov_mat_df = simulationOptions.signal_voxels^simulationOptions.exp; % controls variance of the wishart distribution
simulationOptions.corrs = 0.7^2; %correlations for the covariance matrix from which trial vectors for each stimulus are sampled

% A triple containing the dimensions of one voxel in mm.
simulationOptions.voxelSize_mm = [3 3 3.75];

% A 4-tuple. The first three entries are the x, y and z values for the gaussian
% spatial smoothing kernel FWHM in mm and the fourth is the size of the temporal
% smoothing FWHM.
if sp_smoothing==0
    simulationOptions.spatiotemporalSmoothingFWHM_mm_s = [1 1 1 4.5]; %% %
elseif sp_smoothing==1
    simulationOptions.spatiotemporalSmoothingFWHM_mm_s = [4 4 4 4.5];
end

if big_brain==1
    simulationOptions.brainVol = [64 64 32]; %%
    simulationOptions.effectCen = [20 20 15];
    %where to save the simulation results
    save_path = ['/media/seb/HD_Numba_Juan/sim_results_' num2str(simulationOptions.nf) 'voxel_smoothing' num2str(sp_smoothing) '_noise' num2str(simulationOptions.scannerNoiseLevel) '_' num2str(simulationOptions.nRepititions) 'samps_bigBrain'];
elseif big_brain==0
    simulationOptions.brainVol = simulationOptions.volumeSize_vox*3;
    simulationOptions.effectCen = [randi(round(simulationOptions.volumeSize_vox(1)*1.5)) randi(round(simulationOptions.volumeSize_vox(2)*1.5)) randi(round(simulationOptions.volumeSize_vox(3)*1.5))];
    %where to save the simulation results
    save_path = ['/media/seb/HD_Numba_Juan/sim_results_' num2str(simulationOptions.nf) 'voxel_smoothing' num2str(sp_smoothing) '_noise' num2str(simulationOptions.scannerNoiseLevel) '_' num2str(simulationOptions.nRepititions) 'samps'];
end

disp(save_path)

%% setup the noise levels and collinearity (through trial duration)

% with these two lists, 8 levels will be run in totaldelete(gcp('nocreate'))

trial_duration_list = [2,3,4]; % in seconds
big_sigma_list = [10, 15, 20]; % sigma for the hyper-ellipse containing the mean random vectors
%scan_noise_list = [3000, 5000, 7000, 10000]; %if needed you can add a for
%loop below with different levels of scanner noise


%% run simulations

for trial_duration = 1:length(trial_duration_list)
    for big_sigma = 1:length(big_sigma_list) %for scan_noise = 1:length(scan_noise_list) %
        
        if parjob==0
            simulationOptions.trialDuration = 2; %2;
            simulationOptions.big_sigma = 10; %5^2;
            simulationOptions.scannerNoiseLevel = 10000;
            [accs, DATA] = run_sim_LSS(simulationOptions);delete(gcp('nocreate'))
            disp(simulationOptions.model_list)
            disp(accs)
        else
            simulationOptions.trialDuration = trial_duration_list(trial_duration);
            simulationOptions.big_sigma = big_sigma_list(big_sigma);
            %simulationOptions.scannerNoiseLevel = scan_noise_list(scan_noise);
            
            nworkers = 10;
            all_accs=[];
            SIMS=[];
            parpool(nworkers)
            
            parfor i=1:njobs
                [accs, DATA] = run_sim_LSS(simulationOptions);
                all_accs = [all_accs; accs];
                SIMS = [SIMS, DATA];
            end

            delete(gcp('nocreate'))
            disp(simulationOptions.model_list)
            disp(mean(all_accs))
            
            if exist(save_path,'dir')==0
                mkdir(save_path)
            end
            
            cd(save_path)
            save(['bsig',num2str(simulationOptions.big_sigma),'tsig',num2str(simulationOptions.trial_sigma),'isi', num2str(simulationOptions.trialDuration),'.mat'],'SIMS','all_accs','-v7.3')
            %save(['scan_sig',num2str(simulationOptions.scannerNoiseLevel),'tsig',num2str(simulationOptions.trial_sigma),'isi', num2str(simulationOptions.trialDuration),'.mat'],'SIMS','all_accs')
            
            cd([toolboxRoot '/LSS_project/'])
        end
        
    end
end

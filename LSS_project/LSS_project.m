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

cd ..
pd = pwd; %parent directory for rsatoolbox

%cd /home/seb/Documents/rsatoolbox/
toolboxRoot = pwd; addpath(genpath(toolboxRoot));

% addpath to the libsvm toolbox in case it's not in your matlab path
addpath('/home/seb/Documents/libsvm-3.22/matlab');

%% setup model parameters

parjob = 0; %1 if using cluster, runs njobs
njobs = 50; %number of jobs per noise setting, only if running on many cores

simulationOptions.nRepititions = 20; %4; %repetitions per stimulus
simulationOptions.nruns = 3; %separate fMRI blocks
simulationOptions.stimulusDuration = 1.5; %in seconds

simulationOptions.TR = 1; %scanner TR
simulationOptions.nConditions = 2; %how many stimulus conditions (i.e., classes)?

simulationOptions.svm_options = '-s 0 -t 0 -n 0.5 -h 0'; %nu-svm is supposed to be -s 1??? -t 0 is linear kernel, -t 2 rbf, -h 0 is faster (shrinking heuristic)
% check libsvm website for more information on options
simulationOptions.nf = 100; %number of features (voxels) that go into the classifier

all_models = 1;
if all_models==0
    simulationOptions.model_list = {'B_LSA','B_LSS00','B_LSS01','B_LSS10'};
else
    simulationOptions.model_list = {'B_LSA','B_LSS00','B_LSS01','B_LSS10','B_LSS02','B_LSS11','B_LSS20','B_LSS12','B_LSS21','B_LSS22'};
end

simulationOptions.trial_sigma = 0.5^2; %variance between trials for each stimulus condition
simulationOptions.exp = 1; %modify exponent, higher means more like identity matrix
simulationOptions.volumeSize_vox = [5 5 5]; % size of the signal
simulationOptions.signal_voxels = prod(simulationOptions.volumeSize_vox); % number of signal voxels
simulationOptions.cov_mat_df = simulationOptions.signal_voxels^simulationOptions.exp; % modify exponent, higher means more like identity matrix
simulationOptions.corrs = 0.7^2; %correlations for the covariance matrix

% A triple containing the dimensions of one voxel in mm.
simulationOptions.voxelSize_mm = [3 3 3.75];

% The amount of noise to be added by the simulated scanner. This corresponds to
% the square of the standard deviation of the gaussian distibution from which
% the noise is drawn (?).
simulationOptions.scannerNoiseLevel = 10000; % used to be 3000

% A 4-tuple. The first three entries are the x, y and z values for the gaussian
% spatial smoothing kernel FWHM in mm and the fourth is the size of the temporal
% smoothing FWHM.
simulationOptions.spatiotemporalSmoothingFWHM_mm_s = [4 4 4 4.5];

simulationOptions.brainVol = [64 64 32];
simulationOptions.effectCen = [20 20 15];

%% setup the noise levels and collinearity (through trial duration)

% with these two lists, 8 levels will be run in total

trial_duration_list = [2,3,4,5]; % in seconds
big_sigma_list = [10, 12, 14, 16]; %[2,3,4,5]; % sigma for the hyper-ellipse containing the mean random vectors
%scan_noise_list = [3000, 5000, 7000, 10000]; %if needed you can add a for
%loop below with different levels of scanner noise


%% run simulations

for trial_duration = 1:length(trial_duration_list)
    for big_sigma = 1:length(big_sigma_list) %for scan_noise = 1:length(scan_noise_list) %
        
        if parjob==0
            simulationOptions.trialDuration = 2; %2;
            simulationOptions.big_sigma = 10; %5^2;
            simulationOptions.scannerNoiseLevel = 10000;
            [accs, DATA] = run_sim_LSS(simulationOptions);
            disp(simulationOptions.model_list)
            disp(accs)
            
            cd(pd)
            %save(['sig_',num2str(simulationOptions.big_sigma),'_isi_', num2str(simulationOptions.trialDuration),'.mat'],'accs')
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
            
            if exist([pd '/LSS_project/sim_results/'],'dir')==0
                mkdir([pd '/LSS_project/sim_results/'])
            end
            
            cd([pd '/LSS_project/sim_results/'])
            save(['bsig',num2str(simulationOptions.big_sigma),'tsig',num2str(trial_sigma),'isi', num2str(simulationOptions.trialDuration),'.mat'],'SIMS','all_accs')
            %save(['scan_sig',num2str(simulationOptions.scannerNoiseLevel),'tsig',num2str(trial_sigma),'isi', num2str(simulationOptions.trialDuration),'.mat'],'SIMS','all_accs')
            
            cd([pd '/LSS_project/'])
        end
        
    end
end

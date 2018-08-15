function [varargout] = simulateClusteredfMRIData_fullBrain_LSS2(simulationOptions)
%
% this function simulates fMRI data with a categorical effect inserted at a
% specified location. The remaining regions contain noise that is temporally
% and spatially smooth yet lacks any particular categorical structure.
% the size of the simulated volume is indicated by the volumeSize_vox field
% and the effect center by effectCen. The volume of the simulated effect
% (within the brain volume) is controlled by volumeSize_vox.
% Hamed Nili 2012
%__________________________________________________________________________
% Copyright (C) 2010 Medical Research Council

% Modified by Sebastian Bobadilla Suarez to model voxel covariance in the
% "true signal" and also provide the design matrices for LSS models (cf.
% Mumford et al., 2012)

import rsa.*
import rsa.fig.*
import rsa.fmri.*
import rsa.rdm.*
import rsa.sim.*
import rsa.spm.*
import rsa.stat.*
import rsa.util.*

nNoisyPatterns = nargout - 2;

%% Generate B patterns
nReps = simulationOptions.nRepititions;

%this is the original function for creating clustered activations
%b = generateBetaPatterns(simulationOptions.clusterSpec, prod(simulationOptions.volumeSize_vox));

%this is the new function which models the signal as a m.v. normal
%distribution
b=[];
for rep = 1:nReps*simulationOptions.nConditions %for each stimulus presentation
   rndvec = mvnrnd(simulationOptions.R(mod(rep,simulationOptions.nConditions)+1,:),simulationOptions.cov_mat);% sample each presentation according to previously defined covariance matrix & mean vectors  
   b = [b; rndvec]; 
   sequence(rep) = mod(rep,simulationOptions.nConditions)+1; %start creating the sequence vector as well
end

nVox_wholeBrain = prod(simulationOptions.brainVol);
msk = zeros(simulationOptions.brainVol);
effectCen = simulationOptions.effectCen;
msk([effectCen(1)+1:effectCen(1)+simulationOptions.volumeSize_vox(1)],[effectCen(2)+1:effectCen(2)+simulationOptions.volumeSize_vox(2)],[effectCen(3)+1:effectCen(3)+simulationOptions.volumeSize_vox(3)]) = 1;
B_true = zeros(simulationOptions.nConditions*nReps,nVox_wholeBrain);
B_true(:,find(msk)) = b;
nConditions = size(B_true,1)/nReps;
nVoxels = size(B_true,2);


%% Generate X

%old way of producing the sequence
%sequence = [repmat([1:nConditions],1,simulationOptions.nRepititions),(nConditions+1)*ones(1,ceil(nConditions*simulationOptions.nRepititions/3))];
%sequence = randomlyPermute(sequence);

null_trials = (simulationOptions.nConditions+1)*ones(1,ceil(simulationOptions.nConditions*simulationOptions.nRepititions/3)); %first create the null trials

% alpha = 0.75; %staying probability 
% current_state = binornd(1,0.5) + 1;
% P=[];
% for i = 1:simulationOptions.nConditions*simulationOptions.nRepititions
%     stay_var = binornd(1,alpha);
%     current_state = mod(current_state + stay_var,2)+1; %if stay_var is 1 then we stay
%     P = [P current_state];
% end
%P = randperm(length(sequence)); %get a random permutation on the sequence
P=[];
for i = 1:simulationOptions.nConditions
    P = [P i:simulationOptions.nConditions:length(sequence)];
end

sequence = sequence(P); %permute the sequence
orig_sequence = sequence; %save before adding null trials, will need this below

p2 = randperm(length(sequence)-1); %get random insertion points for null trials
for i = 1:length(null_trials)
    ind = p2(i); %current insertion point for null trial
    sequence = [sequence(1:ind),null_trials(1),sequence(ind+1:end)]; %insert null trial
    null_trials(1)=[]; %deplete the null trial list until finished
end

b = b(P,:); %lets permute our trial-by-trial signal vectors
B_true = B_true(P,:); %as well as their brain-embedded version

nTrials = numel(sequence);
nTRvols = (simulationOptions.trialDuration/simulationOptions.TR)*nTrials;
nSkippedVols = 0;
monitor = 0;
scaleTrialResponseTo1 = 1;

% we first need the saturated design matrix before constructing the LSS
% design matrices and generalizations thereof
simulationOptions.reduce_design=0;
% notice that generateCognitiveModel_fastButTrialsNeedToStartOnVols_LSS
% takes:
% sequence,nTRvols,nSkippedVols,monitor,scaleTrialResponseTo1,simulationOptions
% as arguments whereas the original generateCognitiveModel_fastButTrialsNeedToStartOnVols
% takes:
% sequence,simulationOptions.stimulusDuration*1000,simulationOptions.trialDuration*1000,nTRvols,simulationOptions.TR*1000,nSkippedVols,monitor,scaleTrialResponseTo1
[X,BV_ignore,standardIndexSequence_ignore,hirf_ms_ignore,trialImpulseX_TRvol2] = generateCognitiveModel_fastButTrialsNeedToStartOnVols_LSS(sequence,nTRvols,nSkippedVols,monitor,scaleTrialResponseTo1,simulationOptions);
% it also returns an extra argument (trialImpulseX_TRvol2), which are the
% impulse functions
simulationOptions.reduce_design=1;

%now we can construct the other 9 design matrices with only 5 iterations
%null_inds = p2; %null trial indices
X_all.model0 = X; %save the saturated model
for model = 1:5
    window_length = model; %how many trials at a time would be modelled in the design matrix?
    windows = length(orig_sequence)-(window_length-1); %how many windows fit in the original sequence
    for num_window = 1:windows
        sequence_tmp = orig_sequence; %let's create a new sequence
        msk = 1:length(orig_sequence)>=(num_window+window_length) | 1:length(orig_sequence)<num_window; %mask for anything not in window 
        sequence_tmp(msk) = max(sequence); %let's code out of window with max(sequence) for now
        for null_ind = 1:length(find(sequence==max(sequence))) %for each null trial
            sequence_tmp = [sequence_tmp(1:p2(null_ind)),max(sequence)+1,sequence_tmp(p2(null_ind)+1:end)]; %add in the null trials again for this sequence
        end
        [X_tmp,~,~,~,~] = generateCognitiveModel_fastButTrialsNeedToStartOnVols_LSS(sequence_tmp,nTRvols,nSkippedVols,monitor,scaleTrialResponseTo1,simulationOptions); %create the design matrix
        X_all.(sprintf('model%d',model)).(sprintf('num%d',num_window)) = X_tmp; %save the design matrix
    end
end

nTimePoints = size(X,1);
sig = sqrt(simulationOptions.scannerNoiseLevel);

varargout{1} = B_true;

for o = 1:nNoisyPatterns
	
	%% Generate E matrix
	E = randn(nTimePoints, nVoxels);
    E = sig * E;
    
    %sigma_mat = eye(nTimePoints); % initialize an identity matrix
    %sigma_mat(sigma_mat==0) = simulationOptions.corrs; % add some correlations
    %cov_mat = sigma_mat*sig;
    
    rho = 0.12; %simulationOptions.corrs;
    %produce temporally correlated noise
    V = zeros(nTimePoints,nTimePoints);
    %V = zeros(simulationOptions.nConditions,simulationOptions.nConditions);
    for i = 1:nTimePoints%simulationOptions.nConditions%
        for j = 1:nTimePoints%simulationOptions.nConditions%
            V(i,j) = rho^abs(i-j);
        end
    end
    %cov_mat = sig * V; %V is the correlation matrix that follows rho
    cov_mat = V;
    
    %E2 = mvnrnd(zeros(1,nTimePoints),cov_mat, nVoxels);
    E2 = mvnrnd(E',cov_mat);
    %E2 = mvnrnd(zeros(1,simulationOptions.nConditions),cov_mat, nVoxels);
    %E2 = mvnrnd(zeros(1,nTimePoints),cov_mat, simulationOptions.nConditions);
    %E2 = mvnrnd(zeros(1,simulationOptions.nConditions),cov_mat, nTimePoints);
    %E2 = mvnrnd(B_true',cov_mat);
    %E = E2';
	
    %E2 = repmat(sin([1:nTimePoints]'),1,nVoxels) + 100*randn(nTimePoints,nVoxels);
    
    %B_true = E2; 
    %X = X + E2;
    %E = E + E2'; %probably want to turn off gaussian smoother for this
    
    % Smooth across space
	[E, smoothedYfilename_ignore] = spatiallySmooth4DfMRI_mm(E, simulationOptions.brainVol, simulationOptions.spatiotemporalSmoothingFWHM_mm_s(1:3), simulationOptions.voxelSize_mm);

	% Smooth across time
	E = temporallySmoothTimeSpaceMatrix(E, simulationOptions.spatiotemporalSmoothingFWHM_mm_s(4) / simulationOptions.TR);
    
    % Smooth across time with variants
    %E1 = temporallySmoothTimeSpaceMatrix(E, (simulationOptions.spatiotemporalSmoothingFWHM_mm_s(4) / simulationOptions.TR) + 0);
    %E2 = temporallySmoothTimeSpaceMatrix(E, simulationOptions.spatiotemporalSmoothingFWHM_mm_s(4) / simulationOptions.TR + 1);
    %E3 = temporallySmoothTimeSpaceMatrix(E, (simulationOptions.spatiotemporalSmoothingFWHM_mm_s(4) / simulationOptions.TR) + 2);
    
    %sample from E1, E2, E3
    %tsmooth_inds = crossvalind('Kfold', length(E), 3);
    %E(:,tsmooth_inds==1) = E1(:,tsmooth_inds==1);
    %[E, smoothedYfilename_ignore] = spatiallySmooth4DfMRI_mm(E, simulationOptions.brainVol, simulationOptions.spatiotemporalSmoothingFWHM_mm_s(1:3), simulationOptions.voxelSize_mm);
    %E(:,tsmooth_inds==2) = E2(:,tsmooth_inds==2);
    %[E, smoothedYfilename_ignore] = spatiallySmooth4DfMRI_mm(E, simulationOptions.brainVol, simulationOptions.spatiotemporalSmoothingFWHM_mm_s(1:3), simulationOptions.voxelSize_mm);
    %E(:,tsmooth_inds==3) = E3(:,tsmooth_inds==3);
    
    % Smooth across space after time
    %[E, smoothedYfilename_ignore] = spatiallySmooth4DfMRI_mm(E, simulationOptions.brainVol, simulationOptions.spatiotemporalSmoothingFWHM_mm_s(1:3), simulationOptions.voxelSize_mm);

    

	%% Do GLM for Y_true matrix
	Y_true = X * B_true;
    varargout{2} = msk;
    varargout{3} = Y_true;
	%% Do GLM for Y_noise matrix
	Y_noisy = Y_true;% + E;
    %Y_noisy = (X * (B_true + E2*2)) + E*1; %need to measure correlation of B_true with E2!!!!
    %Y_noisy = ((X * (E2*5)) + (X * B_true))./2 + E*1;
	%B_noisy = inv(X' * X) * X' * Y_noisy;
    
    %fMRI.B_noisy = B_noisy; %used to be saved as just fMRI.B
    fMRI.Y_noisy = Y_noisy;
    %fMRI.X = X;
    fMRI.groundTruth = b;
    
    %save some extra goodies
    %fMRI.E = E;
    fMRI.sequence = sequence;
    %fMRI.Y_true = Y_true;
    fMRI.B_true = B_true;
    %fMRI.B_true = B_true + E2*2;
    %fMRI.B_true = (B_true + E2*5)/2;
    %fMRI.b = b;
    %fMRI.msk = msk;
    %fMRI.saturated_model = trialImpulseX_TRvol2;
    fMRI.volumeSize_vox = simulationOptions.volumeSize_vox;
    fMRI.X_all = X_all;
    fMRI.X_all.model0 = X;
    
	varargout{o + 3} = fMRI;
    
	
	clear E Y_noisy B_noisy;
	
end%for:o

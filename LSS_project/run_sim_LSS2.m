function [err, DATA2] = run_sim_LSS2(simulationOptions)

% all_sigma_est, used to be middle output but only coded for the 1 voxel
% setting

% runs simulations with functions simulateClusteredfMRIData_fullBrain_LSS &
% run_clf_LSS

%Memory footprint can be an issue here...

import rsa.sim.*

%% create covariance matrix

sigma_mat = eye(simulationOptions.signal_voxels); % initialize an identity matrix

R=[];
for cond = 1:simulationOptions.nConditions
    %R = [R; mvnrnd(zeros(1,simulationOptions.signal_voxels),sigma_mat*simulationOptions.big_sigma)]; % create the mean vectors for the true signal
    R = [R; mvnrnd(zeros(1,simulationOptions.signal_voxels),sigma_mat*simulationOptions.big_sigma(cond))];
end

sigma_mat(sigma_mat==0) = simulationOptions.corrs; % add some correlations
cov_mat = wishrnd(sigma_mat*simulationOptions.trial_sigma,round(simulationOptions.cov_mat_df)); % full covariance, depends on df

simulationOptions.R = R; % save the mean vectors
simulationOptions.cov_mat = cov_mat./round(simulationOptions.cov_mat_df); % sigma_mat; % scale the covariance matrix

R_var = var(simulationOptions.R);
[~,rsv_num] = max(R_var); %best signal voxel (only used if simulations' nf = 1)

for run = 1:simulationOptions.nruns
    %simulate fMRI data

    disp('run: ')
    disp(run)

    [~,~,~, DATA.(sprintf('run%d',run)).fMRI] = simulateClusteredfMRIData_fullBrain_LSS2(simulationOptions);

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

    DATA.(sprintf('run%d',run)).tmp_B_true = reshape(DATA.(sprintf('run%d',run)).fMRI.B_true(true_voxel_msk),num_signal_voxels,1);
    DATA2.(sprintf('run%d',run)).var0 = var(DATA.(sprintf('run%d',run)).tmp_B_true);


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
        %DATA.(sprintf('run%d',run)).fMRI.Y_noisy_smoothed = mean(DATA.(sprintf('run%d',run)).fMRI.Y_noisy(:,DATA.rand_signal_voxel-3:DATA.rand_signal_voxel+3)')';
        DATA.(sprintf('run%d',run)).fMRI.Y_noisy2 = DATA.(sprintf('run%d',run)).fMRI.Y_noisy(:,DATA.rand_signal_voxel-3:DATA.rand_signal_voxel+3);
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
    Y_hat = X * DATA.(sprintf('run%d',run)).fMRI.B_LSA;
    %Y_hat_smoothed = X * (inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy_smoothed);
    %residuals = DATA.(sprintf('run%d',run)).fMRI.Y_noisy - Y_hat;
    %DATA.(sprintf('run%d',run)).fMRI.sigma_est = detect_gauss(residuals);
    %B_LSA2 = mean(inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy2,2);
    %DATA.(sprintf('run%d',run)).fMRI.sigma_est = detect_gauss(DATA.(sprintf('run%d',run)).fMRI.Y_noisy);
    DATA2.(sprintf('run%d',run)).tmp_B_LSA = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSA(true_voxel_msk),num_signal_voxels,1);
    [DATA2.(sprintf('run%d',run)).r1,DATA2.(sprintf('run%d',run)).p1] = corr(DATA.(sprintf('run%d',run)).tmp_B_true,DATA2.(sprintf('run%d',run)).tmp_B_LSA);
    r_s(run,i) = DATA2.(sprintf('run%d',run)).r1;
    err(run,i) = immse(DATA.(sprintf('run%d',run)).tmp_B_true,DATA2.(sprintf('run%d',run)).tmp_B_LSA);
    DATA2.(sprintf('run%d',run)).var1 = var(DATA2.(sprintf('run%d',run)).tmp_B_LSA);
    %DATA2.(sprintf('run%d',run)).tmp_B_LSA = tmp_B_LSA;
    X=[];
    %[DATA2.(sprintf('run%d',run)).acf,~,~] = autocorr(DATA.(sprintf('run%d',run)).fMRI.Y_noisy); %sample autocorr estimation
end
model = 'B_LSA';

i=i+1;
% for run = 1:simulationOptions.nruns
%     DATA.(sprintf('run%d',run)).fMRI.(model) = [];
% end

if length(simulationOptions.model_list) > 1

    % LSS00 %one glm per trial (Mumford et al. method)
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.B_LSS00=[];
        for trial = 1:size(refX,2)
            X = DATA.(sprintf('run%d',run)).fMRI.X_all.model1.(sprintf('num%d',trial));
            B_hats = inv(X' * X) * X' * DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
            DATA.(sprintf('run%d',run)).fMRI.B_LSS00 = [DATA.(sprintf('run%d',run)).fMRI.B_LSS00; B_hats(1,:)];
        end
        DATA2.(sprintf('run%d',run)).tmp_B_LSS00 = reshape(DATA.(sprintf('run%d',run)).fMRI.B_LSS00(true_voxel_msk),num_signal_voxels,1);
        [DATA2.(sprintf('run%d',run)).r2,DATA2.(sprintf('run%d',run)).p2] = corr(DATA.(sprintf('run%d',run)).tmp_B_true,DATA2.(sprintf('run%d',run)).tmp_B_LSS00);
        r_s(run,i) = DATA2.(sprintf('run%d',run)).r2;
        err(run,i) = immse(DATA.(sprintf('run%d',run)).tmp_B_true,DATA2.(sprintf('run%d',run)).tmp_B_LSS00);
        DATA2.(sprintf('run%d',run)).var2 = var(DATA2.(sprintf('run%d',run)).tmp_B_LSS00);
        X=[];
    end
    model = 'B_LSS00';

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
        [DATA2.(sprintf('run%d',run)).r3,DATA2.(sprintf('run%d',run)).p3] = corr(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS01);
        r_s(run,i) = DATA2.(sprintf('run%d',run)).r3;
        err(run,i) = immse(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS01);
        DATA2.(sprintf('run%d',run)).var3 = var(tmp_B_LSS01);
        X=[];
    end
    model = 'B_LSS01';

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
        [DATA2.(sprintf('run%d',run)).r4,DATA2.(sprintf('run%d',run)).p4] = corr(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS10);
        r_s(run,i) = DATA2.(sprintf('run%d',run)).r4;
        err(run,i) = immse(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS10);
        DATA2.(sprintf('run%d',run)).var4 = var(tmp_B_LSS10);
        X=[];
    end
    model = 'B_LSS10';

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
        [DATA2.(sprintf('run%d',run)).r5,DATA2.(sprintf('run%d',run)).p5] = corr(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS02);
        r_s(run,i) = DATA2.(sprintf('run%d',run)).r5;
        err(run,i) = immse(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS02);
        DATA2.(sprintf('run%d',run)).var5 = var(tmp_B_LSS02);
        X=[];
    end
    model = 'B_LSS02';

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
        [DATA2.(sprintf('run%d',run)).r6,DATA2.(sprintf('run%d',run)).p6] = corr(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS11);
        r_s(run,i) = DATA2.(sprintf('run%d',run)).r6;
        err(run,i) = immse(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS11);
        DATA2.(sprintf('run%d',run)).var6 = var(tmp_B_LSS11);
        X=[];
    end
    model = 'B_LSS11';

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
        [DATA2.(sprintf('run%d',run)).r7,DATA2.(sprintf('run%d',run)).p7] = corr(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS20);
        r_s(run,i) = DATA2.(sprintf('run%d',run)).r7;
        err(run,i) = immse(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS20);
        DATA2.(sprintf('run%d',run)).var7 = var(tmp_B_LSS20);
        X=[];
    end
    model = 'B_LSS20';

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
        [DATA2.(sprintf('run%d',run)).r8,DATA2.(sprintf('run%d',run)).p8] = corr(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS12);
        r_s(run,i) = DATA2.(sprintf('run%d',run)).r8;
        err(run,i) = immse(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS12);
        DATA2.(sprintf('run%d',run)).var8 = var(tmp_B_LSS12);
        X=[];
    end
    model = 'B_LSS12';

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
        [DATA2.(sprintf('run%d',run)).r9,DATA2.(sprintf('run%d',run)).p9] = corr(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS21);
        r_s(run,i) = DATA2.(sprintf('run%d',run)).r9;
        err(run,i) = immse(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS21);
        DATA2.(sprintf('run%d',run)).var9 = var(tmp_B_LSS21);
        X=[];
    end
    model = 'B_LSS21';

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
        [DATA2.(sprintf('run%d',run)).r10,DATA2.(sprintf('run%d',run)).p10] = corr(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS22);
        r_s(run,i) = DATA2.(sprintf('run%d',run)).r10;
        err(run,i) = immse(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_LSS22);
        DATA2.(sprintf('run%d',run)).var10 = var(tmp_B_LSS22);
        X=[];
    end

    i = i+1;
    for run = 1:simulationOptions.nruns
        DATA.(sprintf('run%d',run)).fMRI.(model) = []; %clear variable due to memory overload
    end

end

if simulationOptions.ridge == 1
    % LSSridge %penalty term is lambda*(B-S)'*(B-S)
    prior = 'B_LSS00';
    %prior = 'B_LSA';
    for test_run = 1:simulationOptions.nruns
        train_run = mod(run,2)+1;
        targetB = DATA2.(sprintf('run%d',train_run)).tmp_B_LSA;
        %targetB = DATA2.(sprintf('run%d',train_run)).tmp_B_LSS00;
        X_test = DATA.(sprintf('run%d',test_run)).fMRI.X_all.model0; %uses LSA design matrix & LSS00 betas
        X_train = DATA.(sprintf('run%d',train_run)).fMRI.X_all.model0; %train_run?
        %fun = @(lambda)ridge_estimator(lambda,X0,DATA.(sprintf('run%d',train_run)).fMRI.Y_noisy,DATA.(sprintf('run%d',train_run)).fMRI.(prior),true_voxel_msk,num_signal_voxels,targetB);
        Y_train = DATA.(sprintf('run%d',train_run)).fMRI.Y_noisy;
        Y_test = DATA.(sprintf('run%d',test_run)).fMRI.Y_noisy;
        prior_betas = DATA.(sprintf('run%d',train_run)).fMRI.(prior);
        fun = @(lambda)ridge_estimator(lambda,X_train,X_test,Y_train,Y_test,prior_betas,true_voxel_msk,num_signal_voxels,targetB);
        lambda = fminbnd(fun,0,100);
        %lambda = 1000000;
        DATA.(sprintf('run%d',train_run)).fMRI.B_ridge = inv(X_train'*X_train + lambda*eye(size(refX,2)))*(X_train'*Y_train + lambda*prior_betas);
        tmp_B_ridge = reshape(DATA.(sprintf('run%d',train_run)).fMRI.B_ridge(true_voxel_msk),num_signal_voxels,1);
        [DATA2.(sprintf('run%d',train_run)).r11,DATA2.(sprintf('run%d',train_run)).p11] = corr(DATA.(sprintf('run%d',train_run)).tmp_B_true,tmp_B_ridge);
        r_s(train_run,i) = DATA2.(sprintf('run%d',train_run)).r11;
        err(train_run,i) = immse(DATA.(sprintf('run%d',train_run)).tmp_B_true,tmp_B_ridge);
        DATA2.(sprintf('run%d',train_run)).var11 = var(tmp_B_ridge);
        X=[];
    end
    model = 'B_ridge';
elseif   simulationOptions.ridge == 2
    prior = 'B_LSS00';
    for run = 1:simulationOptions.nruns
        DATA2.(sprintf('run%d',run)).r11=[];
        DATA2.(sprintf('run%d',run)).p11=[];
        DATA2.(sprintf('run%d',run)).var11=[];
        for j = 1:length(simulationOptions.lambda_list)
            lambda = simulationOptions.lambda_list(j);
            X = DATA.(sprintf('run%d',run)).fMRI.X_all.model0;
            Y = DATA.(sprintf('run%d',run)).fMRI.Y_noisy;
            prior_betas = DATA.(sprintf('run%d',run)).fMRI.(prior);
            DATA.(sprintf('run%d',run)).fMRI.B_ridge.(sprintf('lambda%d',i))= inv(X'*X + lambda*eye(size(refX,2)))*(X'*Y + lambda*prior_betas); %ridge with prior
            %DATA.(sprintf('run%d',run)).fMRI.B_ridge.(sprintf('lambda%d',i)) = inv(X'*X + lambda*eye(size(refX,2)))*(X'*Y); %normal ridge
            tmp_B_ridge = DATA.(sprintf('run%d',run)).fMRI.B_ridge.(sprintf('lambda%d',i));
            tmp_B_ridge = reshape(tmp_B_ridge(true_voxel_msk),num_signal_voxels,1);
            [DATA2.(sprintf('run%d',run)).r11(j),DATA2.(sprintf('run%d',run)).p11(j)] = corr(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_ridge);
            r_s(run,i+j-1) = DATA2.(sprintf('run%d',run)).r11(j);
            err(run,i+j-1) = immse(DATA.(sprintf('run%d',run)).tmp_B_true,tmp_B_ridge);
            DATA2.(sprintf('run%d',run)).var11(j) = var(tmp_B_ridge);
            X=[];
        end
    end
    model = 'B_ridge';
end

for run = 1:simulationOptions.nruns
    DATA.(sprintf('run%d',run)).fMRI.(model) = []; %clear variable due to memory overload
end


for run = 1:simulationOptions.nruns
    DATA.(sprintf('run%d',run)).fMRI.B_LSS00=[]; %since padding finished on all models, clear up more memory


    %DATA2.(sprintf('run%d',run)).X = DATA.(sprintf('run%d',run)).fMRI.X;
    %DATA2.(sprintf('run%d',run)).X_all = DATA.(sprintf('run%d',run)).fMRI.X_all;
    DATA2.(sprintf('run%d',run)).groundTruth = DATA.(sprintf('run%d',run)).fMRI.groundTruth;
    DATA2.(sprintf('run%d',run)).sequence = DATA.(sprintf('run%d',run)).fMRI.sequence;
    %DATA2.(sprintf('run%d',run)).b = DATA.(sprintf('run%d',run)).fMRI.b;
    DATA2.(sprintf('run%d',run)).volumeSize_vox = DATA.(sprintf('run%d',run)).fMRI.volumeSize_vox;
    DATA2.(sprintf('run%d',run)).simulationOptions = simulationOptions;
    %DATA2.(sprintf('run%d',run)).sigma_est = DATA.(sprintf('run%d',run)).fMRI.sigma_est;

    %all_sigma_est(run) = DATA.(sprintf('run%d',run)).fMRI.sigma_est;
end


end

function sq_error = ridge_estimator(lambda,X_train,X_test,Y_train,Y_test,S,true_voxel_msk,num_signal_voxels,targetB)
    X = X_train;
    ridge_betas = inv(X'*X + lambda*eye(size(X,2)))*(X'*Y_train + lambda*S);
    Y_hat = X_test*ridge_betas;
    tmp_B_ridge = reshape(ridge_betas(true_voxel_msk),num_signal_voxels,1);
    %sq_error = immse(Y,Y_hat);
    sq_error = immse(Y_test,Y_hat);
    %sq_error = immse(targetB,tmp_B_ridge);
end

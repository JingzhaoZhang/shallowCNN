%% default configuration
setup_yang
mopts=struct('poolType','compact_TS', ... % could specify different pooling types
             ...% available options: bilinear, compact_RM, compact_TS, fisher, fcfc
             'use448', false, ... % input image resize to 448 (true) or 224 (false)
             ... % only for compact_RM & compact_TS
             'projDim', 8192,... % the projection dimension for compact features
             'learnW', false, ... % whether to learn the random weights in the compact layer
             ... % only for pretrain multi-layer percepturns
             'isPretrainFCs', false,... % true to train multilayer perception
             'fc1Num', -1,... % the number of neurons in the 1st hidden layer
             'fc2Num', -1,... % the number of neurons in the 2nd hidden layer
             'fcInputDim', 512*512,... % input feature dimension
             ... % after training the above multilayer perception, use this option
             ... % to specify the trained model location, this will concats to the 
             ... % network that has already been set up. 
             'nonLinearClassify', '', ... 
             ... % some options that usually don't change
             'ftConvOnly', false,... % if true, set lr of the classification layer to 0. 
             'classifyType', 'LR',...% 'SVM', 'LR': the loss used in the classification layer
             'initMethod', 'random'); % 'pretrain', 'random': when set to 
                % pretrain, extract activation first and call vl_svmtrain
                % to initialize the parameters of the final classification
                % layer. When set to random, use Xavier initialization. 

% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='MIT_Places';
% network: 'VGG_M', 'VGG_16'
network='VGG_M';
% in the case of multiple gpus on a system, specify which one to use. 
gpuId=1;
% tag to diffrentiate multiple versions of experiments. 
tag='some_tag';
% Save the trained model every saveIter epoches. Mainly for saving disk space.
saveInter=1;

% batch size, usually VGG_M=32; VGG_16=8;
batchSize=32;
% set a reasonable learning rate
learningRate=[0.1 0.01 0.001];
% weightDecay: usually use the default value
weightDecay=0.0005;

addpath no_publish/mit_places205/
%% change the options based on default value
gpuId=1;
tag='camera_ready';
batchSize=32;
mopts.poolType='bilinear';
mopts.use448=false;
run_one_experiment_places(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);
%% change the options based on default value
gpuId=3;
tag='camera_ready';
batchSize=32;
mopts.poolType='compact_TS';
mopts.use448=false;
run_one_experiment_places(dataset,network, gpuId, tag, saveInter,...
                       batchSize, learningRate, weightDecay, mopts);                   
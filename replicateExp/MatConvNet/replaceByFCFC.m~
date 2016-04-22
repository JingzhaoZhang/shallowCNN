%% default configuration
setup_yang
mopts=struct('poolType','compact_TS', ... % could specify different pooling types
             ...% available options: bilinear, compact_RM, compact_TS, fisher, fcfc
             'use448', true, ... % input image resize to 448 (true) or 224 (false)
             ... % only for compact_RM & compact_TS
             'projDim', 8192,... % the projection dimension for compact features
             'learnW', true, ... % whether to learn the random weights in the compact layer
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
             'initMethod', 'pretrain'); % 'pretrain', 'random': when set to 
                % pretrain, extract activation first and call vl_svmtrain
                % to initialize the parameters of the final classification
                % layer. When set to random, use Xavier initialization. 

% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='CUB';
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
learningRate=[0 ones(1, 30)*1E-3];
% weightDecay: usually use the default value
weightDecay=0.0005;

%% change the options based on default values
gpuId=1;
dataset='CUB';
network='VGG_M';
saveInter=1;
batchSize = 32;
tags={'jingzhao_Compact2ReplaceWithFCFC', '_jingzhao_averageReplaceWithFCFC'};
filenames = {'CUB_VGG_M_LR_compact2_8192_fixW_224_final_1', 'CUB_VGG_M_LR_average_224_jingzhao'};
epochs = [89, 30];

endLayer = 13;
lastNchannel = 512;
mopts.use448=false;
mopts.poolType='fcfc';


for i = 1:numel(epochs)
    
    filename = filenames(i);
    epoch = epochs(i);
    tag = tags{i};
    netlink = strcat({'data/exp_yang_fv/'}, filename, {'/net-epoch-'}, {num2str(epoch)}, {'.mat'});
    net = load(netlink{1});
    net = net.net;
    %bilinear, compact_RM, compact_TS, fisher, fcfc
    run_one_experiment_jz(dataset,network, net, endLayer, lastNchannel, gpuId, tag, saveInter,...
                        batchSize, learningRate, weightDecay, mopts);
end

%% change the options based on default values
gpuId=3;
dataset='CUB';
network='VGG_16';
saveInter=1;
batchSize = 8;
tags={'_jingzhao_averageReplaceWithFCFC', '_jingzhao_compact2ReplaceWithFCFC'};
filenames = {'CUB_VGG_16_LR_average_224_jingzhao', 'CUB_VGG_16_LR_compact2_8192_fixW_224_final_1'};
epochs = [19, 16];

endLayer = 29;
lastNchannel = 512;
mopts.use448=false;
mopts.poolType='fcfc';


for i = 2:numel(epochs)
    learningRate = learningRate * 0.1;
    filename = filenames(i);
    epoch = epochs(i);
    tag = tags{i};
    netlink = strcat({'data/exp_yang_fv/'}, filename, {'/net-epoch-'}, {num2str(epoch)}, {'.mat'});
    net = load(netlink{1});
    net = net.net;
    %bilinear, compact_RM, compact_TS, fisher, fcfc
    run_one_experiment_jz(dataset,network, net, endLayer, lastNchannel, gpuId, tag, saveInter,...
                        batchSize, learningRate, weightDecay, mopts);
end

 
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
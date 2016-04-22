%% Setup mopts
setup_yang
mopts=struct('poolType','average', ... % could specify different pooling types
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
    'initMethod', 'pretrain'); % 'pretrain', 'random': when set to
% pretrain, extract activation first and call vl_svmtrain
% to initialize the parameters of the final classification
% layer. When set to random, use Xavier initialization.

learningRate=[ones(1, 15)*1E-2 ones(1, 15)*1E-3];
% weightDecay: usually use the default value
weightDecay=0.0005;

%% Setup net
gpuId=4;
dataset='CUB';
network='VGG_M';
saveInter=1;
batchSize = 32;
tag='jingzhao_sigmoid';
filename = 'CUB_VGG_M_LR_average_224_jingzhao';
epoch = 30;




netlink = ['data/exp_yang_fv/', filename, '/net-epoch-', num2str(epoch), '.mat'];
net = load(netlink);
net = net.net;

net.layers = net.layers(1:end-1);
net.layers{end+1}=struct('type', 'loss',...
    'name', 'final_loss', ...
    'lossType', 'sigmoid');

%% some parameters should be tuned
opts.train.batchSize = batchSize;
opts.train.learningRate = learningRate;
opts.train.weightDecay = weightDecay;
opts.train.momentum = 0.9 ;

% set the batchSize of initialization
mopts.batchSize=opts.train.batchSize;

% other parameters that usually is fixed
opts.imdbPath = consts(dataset, 'imdb');
opts.train.expDir = consts(dataset, 'expDirOne', 'cfgId', mopts.poolType, 'learnW', mopts.learnW,...
    'projDim', mopts.projDim, 'network', network, ...
    'pretrainMethod', mopts.classifyType, 'use448', mopts.use448, 'tag', tag);
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = gpuId ;
%gpuDevice(opts.train.gpus); % don't want clear the memory
opts.train.prefetch = true ;
opts.train.sync = false ; % for speed
opts.train.cudnn = true ; % for speed
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts.train.saveInter=saveInter;
imdb = load(opts.imdbPath) ;
% in case some dataset only has val/test
opts.train.val=union(find(imdb.images.set==2), find(imdb.images.set==3));
opts.train.train=[];

% when using few shots
valInter=1;



bopts=net.normalization;
%bopts.transformation = 'stretch' ; %TODO (need such augmentation?)
bopts.numThreads = 12;
fn = getBatchWrapper(bopts) ;



opts.train.backPropDepth=inf; % could limit the backprop
[net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', true, 'valInter', valInter);








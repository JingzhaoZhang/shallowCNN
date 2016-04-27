function autoSelect()
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

learningRate=[ ones(1, 30)*1E-3];
% weightDecay: usually use the default value
weightDecay=0.0005;

%% Setup net
twoEpoch = 0;
twoLayer = 30;
gpuId=4;
dataset='CUB';
network='VGG_16';
saveInter=1;
batchSize = 32;
tag='jingzhao_autoselect';



net = load(consts(dataset, network));
net.layers = net.layers(1:35);

% initFCparam = ones(5, 5, 1, 512, 'single')/25;
% 
% net.layers{end+1} = struct('type', 'conv', 'name', 'flexiblePool', ...
%     'weights', {{initFCparam, zeros(512, 1, 'single')}}, ...
%     'stride', 1, ...
%     'pad', [2,1,2,1], ...
%     'learningRate', [1 2]);
% net.layers{end+1}=struct('type', 'relu', 'name', 'relu6');
% 
% 
% net.layers{end+1}=struct('type', 'pool',...
%     'method', 'avg', 'pool', 13, ...
%     'name', 'avg_pool', 'pad', 0, 'stride', 1);



net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);


net.addLayer('maxpool', ...
    dagnn.Pooling('poolSize', [5, 5], 'pad',[2, 2, 2, 2]),...
    'x30', 'maxOut');

net.addLayer('maxpoolsqrt', ...
    Sqrt(), 'maxOut', 'maxSqrtOut');

net.addLayer('maxpoolL2', ...
    L2Norm('dimension', 2), 'maxSqrtOut', 'maxL2Out');

net.addLayer('compactPool', ...
    yang_Compact_TS_2stream('previousChannels', [512, 512], 'outDim', 4096),...
    {'x30', 'x30'}, 'compactOut');

net.addLayer('compactsqrt', ...
    Sqrt(), 'compactOut', 'compactSqrtOut');

net.addLayer('compactL2', ...
    L2Norm('dimension', 2), 'compactSqrtOut', 'compactL2Out');


net.addLayer('fcfcsqrt', ...
    Sqrt(), 'x35', 'fcfcSqrtOut');

net.addLayer('fcfcL2', ...
    L2Norm('dimension', 2), 'fcfcSqrtOut', 'fcfcL2Out');

net.addLayer('concat', ...
    dagnn.Concat('dim', 3, 'numInputs', 3), ...
    {'maxL2Out', 'compactL2Out', 'fcfcL2Out'}, 'concatOut');


num_classes=consts(dataset, 'num_classes');

net.addLayer('classify', ...
    dagnn.Conv('size', [1 1 512+4096+4096, num_classes], 'pad', 0, ...
    'stride', 1), ...
    'concatOut', 'prediction', {'classifyf', 'classifyb'})

w1 = init_weight('xavierimproved', 1, 1, 512+4096+4096, num_classes, 'single');
f = net.getParamIndex('classifyf');
net.params(f).value = w1;
net.params(f).learningRate = 1;
net.params(f).weightDecay = 1;
f = net.getParamIndex('classifyb');
net.params(f).value = zeros(1,1,num_classes, 'single');
net.params(f).learningRate = 1;
net.params(f).weightDecay = 1;

net.addLayer('loss',...
    dagnn.Loss('loss', 'softmaxlog'), ...
    'prediction', 'prob')

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
%opts.train.sync = false ; % for speed
%opts.train.cudnn = true ; % for speed
opts.train.numEpochs = numel(opts.train.learningRate) ;
%opts.train.saveInter=saveInter;
imdb = load(opts.imdbPath) ;
% in case some dataset only has val/test
opts.train.val=union(find(imdb.images.set==2), find(imdb.images.set==3));
opts.train.train=[];

% when using few shots
valInter=1;



%bopts=net.normalization;
%bopts.transformation = 'stretch' ; %TODO (need such augmentation?)
bopts.numThreads = 12;
fn = getBatchWrapper(bopts) ;

train = find(imdb.images.set==1);
val = find(imdb.images.set==2);
%opts.train.backPropDepth=inf; % could limit the backprop
info = cnn_train_dag(net, imdb, fn, opts.train, 'train', train, 'val', val, struct('gpus', [gpuId]))
%[net,info] = cnntrain_jz(net, imdb, fn, twoEpoch, twoLayer, opts.train, 'conserveMemory', true, 'valInter', valInter);
end


function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.
    switch lower(opts)
      case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
      case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
      case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
      otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
    end
end

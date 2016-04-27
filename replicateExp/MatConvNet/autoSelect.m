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
tag='jingzhao_flexiblePoolL2';



net = load(consts(dataset, network));
net.layers = net.layers(1:30);


initFCparam = ones(5, 5, 1, 512, 'single')/25;

net.layers{end+1} = struct('type', 'conv', 'name', 'flexiblePool', ...
    'weights', {{initFCparam, zeros(512, 1, 'single')}}, ...
    'stride', 1, ...
    'pad', [2,1,2,1], ...
    'learningRate', [1 2]);
net.layers{end+1}=struct('type', 'relu', 'name', 'relu6');


net.layers{end+1}=struct('type', 'pool',...
    'method', 'avg', 'pool', 13, ...
    'name', 'avg_pool', 'pad', 0, 'stride', 1);

net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);





num_classes=consts(dataset, 'num_classes');

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
[net,info] = cnntrain_jz(net, imdb, fn, twoEpoch, twoLayer, opts.train, 'conserveMemory', true, 'valInter', valInter);
end


function net=addClassification(net, lossType, initFCparam)
    % convert the 'SVM' and 'LR' to internal representation
    if strcmp(lossType, 'LR')
        lossType='softmaxlog';
    elseif strcmp(lossType, 'SVM')
        lossType='mhinge';
    else
        error('unknown loss type');
    end

    net=addFC(net, 'fc_final', initFCparam, 'specified');
    
    % add the last softmax classification layer
    net.layers{end+1}=struct('type', 'loss',...
                             'name', 'final_loss', ...
                             'lossType', lossType);
end

function initFCparam=getFCinitWeight(...
         initMethod,...
         nfeature, nclass,...
         classificationType, network, cfgId, dataset, netBeforeClassification,...
         use448, batchSize,tag)
    
    if strcmp(initMethod, 'random')
        % random initialize
        initFCparam={{init_weight('xavierimproved', 1, 1, nfeature, nclass, 'single'),...
                    zeros(nclass, 1, 'single')}};
    elseif strcmp(initMethod, 'pretrain')
        weight_file=consts(dataset, 'classifierW',...
            'pretrainMethod', classificationType, ...
            'network', network, ...
            'cfgId', cfgId, ...
            'projDim', nfeature, ...
            'use448', use448);
        weight_file = [weight_file(1:end-4), tag, '.mat'];
        if exist(weight_file, 'file') == 2
            % svm or logistic initialized weight, load from disk
            load(weight_file);
        else
            % get activations from the last conv layer % checkpoint
            [trainFV, trainY, valFV, valY]=...
                get_activations_dataset_network_layer(...
                    dataset, network, cfgId, use448, netBeforeClassification, batchSize, tag);
            % train SVM or LR weight, and test it on the validation set. 
            [w, b, acc, map, scores]= train_test_vlfeat(classificationType, ...
                squeeze(trainFV), squeeze(trainY), squeeze(valFV), squeeze(valY));
            % reshape the parameters to the input format
            w=reshape(single(w), 1, 1, size(w, 1), size(w, 2));
            b=single(squeeze(b));
            initFCparam={{w, b}};
            % save on disk
            savefast(weight_file, 'initFCparam', 'acc', 'map', 'scores');
        end
        % end of using pretrain weight
    else
        error('init method unknown');
    end
end

function net=addSqrt(net)
    net.layers{end+1}=struct('type', 'custom',...
        'forward',  @yang_sqrt_forward, ...
        'backward', @yang_sqrt_backward, ...
        'name', 'sign_sqrt');
end

function net=addL2norm(net)
    % implement my own layer, much faster
    net.layers{end+1}=struct('type', 'custom',...
        'forward',  @yang_l2norm_forward, ...
        'backward', @yang_l2norm_backward, ...
        'name', 'L2_normalize');
end

function net=addFC(net, name, initFCparam, initMethod)
    if strcmp(initMethod, 'random')
        initFCparam={{init_weight('xavierimproved', 1, 1, initFCparam(1), initFCparam(2), 'single'),...
                      zeros(initFCparam(2), 1, 'single')}};
    elseif strcmp(initMethod, 'specified')
        % nothing should be done
    else
        error('In addFC, unknown parameter initialization method.');
    end
    
    net.layers{end+1} = struct('type', 'conv', 'name', name, ...
       'weights', initFCparam, ...
       'stride', 1, ...
       'pad', 0, ...
       'learningRate', [1 2]);
end




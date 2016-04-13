function run_one_experiment(dataset, network, gpuId, tag,...
            saveInter, batchSize, learningRate, weightDecay, varargin)
    prepare_dataset(dataset);

    setup_yang;
    gpuDevice(gpuId);
    
    mopts.poolType='bilinear';
    % only for compact_bilinear
    mopts.projDim=8192;
    mopts.learnW=false;
    % only for pretrain multilayer perception
    mopts.isPretrainFCs=false;
    mopts.fc1Num=-1;
    mopts.fc2Num=-1;
    mopts.fcInputDim=512*512;

    % some usually fixed params
    mopts.ftConvOnly=false;
    mopts.use448=true;
    mopts.classifyType='LR'; % or SVM
    mopts.initMethod='pretrain'; % or 'random'
    mopts.nonLinearClassify='';
    
    mopts = vl_argparse(mopts, varargin);
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
    
    valInter=1;

    % initialize network 
    net=modifyNetwork(network, dataset, mopts);
    vl_simplenn_display(net);

    if mopts.use448 && ~mopts.isPretrainFCs
        net.normalization.averageImage=imresize(net.normalization.averageImage, ...
            448.0 / size(net.normalization.averageImage, 1));
        net.normalization.imageSize=[448 448 3];
    end

    bopts=net.normalization;
    %bopts.transformation = 'stretch' ; %TODO (need such augmentation?)
    bopts.numThreads = 12;
    fn = getBatchWrapper(bopts) ;
    
    %% learn the full model
    assert(strcmp(net.layers{end-1}.name, 'fc_final'));
    %net.layers{end-1}.learningRate=[0, 0];
    opts.train.backPropDepth=2; % could limit the backprop
    [net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', true, 'valInter', valInter);
end

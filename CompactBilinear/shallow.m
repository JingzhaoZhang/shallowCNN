%% CUB_VGG_16_LR_compact2_8192_fixW_448_final_1 

% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='CUB';
network='VGG_16';
batchSize = 8;
endlayer1 = 30;
endlayer2 = 33;

save_activations = true;


use448 = true;
filename = 'CUB_VGG_16_LR_compact2_8192_fixW_448';
net = load(['shallow_models/', filename '.mat']);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations)

%% MIT_VGG_16_LR_compact2_8192_fixW_448_final

% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='MIT';
network='VGG_16';
batchSize = 8;
use448 = true;
endlayer1 = 30;
endlayer2 = 33;

save_activations = true;

filename = 'MIT_VGG_16_LR_compact2_8192_fixW_448_final';
net = load(['shallow_models/', filename '.mat']);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations)

%% CUB_VGG_M pretrain

% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='CUB';
network='VGG_M';
batchSize = 32;
endlayer1 = 14;
endlayer2 = 19;

save_activations = true;


use448 = true;
filename = 'CUB_VGG_M_pretrain';
net = load(['data/models/imagenet-vgg-m.mat']);
%net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations)

%% MIT_VGG_M pretrain

% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='MIT';
network='VGG_M';
batchSize = 32;
endlayer1 = 14;
endlayer2 = 19;

save_activations = true;


use448 = true;
filename = 'MIT_VGG_M_pretrain';
net = load(['data/models/imagenet-vgg-m.mat']);
%net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations)

%% MIT VGG_M MULTI Epoch


% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='MIT';
network='VGG_M';
batchSize = 32;
endlayer1 = 14;
endlayer2 = 17;

save_activations = true;


use448 = true;
filename = 'MIT_VGG_M_LR_compact_TS_8192_learnW_448_jingzhao';

epochs = [1, 2, 3, 5, 10, 20];
epochs = 30;

for i = 1:numel(epochs)
epoch = epochs(i);

net = load(['shallow_models/exp_yang_fv/' filename '/net-epoch-' num2str(epoch) '.mat']);
net = net.net;

compare_shallow_regular([filename '_epoch_' num2str(epoch)],...
    dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations)


end

%% MIT_VGG_16_LR_fcfc_224_test_batch32
% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='MIT';
network='VGG_16';
batchSize = 8;
use448 = false;
endlayer1 = 30;
endlayer2 = 35;

save_activations = true;

filename = 'MIT_VGG_16_LR_fcfc_224_test_batch32';
net = load(['/x/jingzhao/CompactBilinear/shallow_models/MIT_VGG_16_LR_fcfc_224_test_batch32/net-epoch-60']);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations)




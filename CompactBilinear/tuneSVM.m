%% CUB_VGG_M pretrain

% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='CUB';
network='VGG_M';
batchSize = 32;
endlayer1 = 14;
endlayer2 = 19;

save_activations = true;

% SVMopts.epsilon, reg_coeff, maxiter, method1, method2
SVMopts.epsilon = 1E-4;
SVMopts.reg_coeff = 0.0001;
SVMopts.maxiter = 1E8;
SVMopts.method1 = 'LR';
SVMopts.method2 = 'LR';

% SVM
% 1E-8 0.001 0.515
% 1E-8 0.01 0.515
% 1E-8 0.1 0.52
% 1E-8 1 0.53
% 1E-8 10 0.43
% 1E-5 1 0.531
% 1E-5 10 0.44
% 1E-5 0.1 0.52
% 1E-5 0.01 0.515
%           Use this if LR does not converge         1E-4 1 0.543
% 1E-4 10 0.44
% 1E-4 0.1 0.525
% 1E-3 1 0.503
% 1E-3 0.1 0.538
% 1E-3 0.01 0.521

% LR
% 1E-3 0.01 0.546
% 1E-3 0.1 0.554
%                          1E-3 1 0.556
% 1E-3 10 0.549
% 1E-2 10 0.517
% 1E-2 1 0.518
% 1E-2 0.1 0.515
% 1E-4 0.1 0.538
% 1E-4 0.001 0.541
% 1E-4 0.01 0.541
% 1E-4 0.0001 0.543

use448 = false;
filename = 'CUB_VGG_M_pretrain';
net = load(['data/models/imagenet-vgg-m.mat']);
%net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1,...
    endlayer2, use448, net, batchSize, save_activations, SVMopts)


%% MIT VGGM FCFC

% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='MIT';
network='VGG_M';
batchSize = 32;
endlayer1 = 14;
endlayer2 = 19;

save_activations = true;

% SVMopts.epsilon, reg_coeff, maxiter, method1, method2
SVMopts.epsilon = 1E-4;
SVMopts.reg_coeff = 1;
SVMopts.maxiter = 1E6;
SVMopts.method1 = 'Stop';
SVMopts.method2 = 'SVM';

% LR
% 1E-5 1E-3 61.1
% 1E-4 1E-3 61.3
% 1E-3 1E-3 61.4
% 1E-2 1E-3 60.7
% 1E-6 1E-2 61.5
% 1E-5 1E-2 62.0
% 1E-4 1E-2 61.6
% 1E-3 1E-2 60.9
% 1E-7 1E-1 61.9
% 1E-6 1E-1 62.4
% 1E-5 1E-1 61.5
% 1E-7 1 62.4
% 1E-8 1 62.6
% 1E-9 1 62.7
% 1E-10 1 62.8
% 1E-11 62.4
% 1E-11 10 62.4
% 1E-10 10 63
% 1E-9 10 63
% 1E-8 10 62.6
% 1E-11 100 0.631
% 1E-11 1000 0.634
% 1E-11 10000 0.639
% 1E-11 100000 0.645
% 1E-11 1000000 0.650
% 1E-10 1E7 0.653
% 1E-11 1E7 0.653
% 1E-12 1E7 0.653
% 1E-11 100000000 0.615
use448 = false;
filename = 'MIT_VGG_M_fcfc_finetuneSVM';
net = load('/mnt/x/yang/exp_data/fv_layer/exp_yang_fv/MIT_VGG_M_LR_fcfc_224_final_1/net-epoch-20.mat');
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1,...
    endlayer2, use448, net, batchSize, save_activations, SVMopts)
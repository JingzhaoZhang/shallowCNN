% SVMopts.epsilon, reg_coeff, maxiter, method1, method2
% SVMopts.epsilon = 1E-3;
% SVMopts.reg_coeff = 1;
% SVMopts.maxiter = 1E6;
% SVMopts.method1 = 'LR';
% SVMopts.method2 = 'LR';

% % SVMopts.epsilon, reg_coeff, maxiter, method1, method2
SVMopts.epsilon = 1E-4;
SVMopts.reg_coeff = 1;
SVMopts.maxiter = 1E6;
SVMopts.method1 = 'SVM';
SVMopts.method2 = 'SVM';

%% MIT VGG_16 FCFC
% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='MIT';
network='VGG_16';
batchSize = 8;
use448 = false;
endlayer1 = 30;
endlayer2 = 35;

save_activations = true;

filename = 'MIT_VGG_16_LR_fcfc_224_test_batch32';
epoch = 60;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)
%% CUB VGG_16 FCFC
% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='CUB';
network='VGG_16';
batchSize = 8;
use448 = false;
endlayer1 = 30;
endlayer2 = 35;

save_activations = true;

filename = 'CUB_VGG_16_LR_fcfc_224_final_1';
epoch = 26;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)

%% CUB VGG_M FCFC
% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='CUB';
network='VGG_M';
batchSize = 32;
use448 = false;
endlayer1 = 14;
endlayer2 = 19;

save_activations = true;

filename = 'CUB_VGG_M_LR_fcfc_224_final_1';
epoch = 13;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)

%% MIT VGG_M FCFC
% dataset: 'CUB', 'MIT', 'FMD', 'DTD'
dataset='MIT';
network='VGG_M';
batchSize = 32;
use448 = false;
endlayer1 = 14;
endlayer2 = 19;

save_activations = true;

filename = 'MIT_VGG_M_LR_fcfc_224_final_1';
epoch = 20;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)


%% MIT VGGM bilinear average compactTS

dataset='MIT';
network='VGG_M';
batchSize = 32;
use448 = true;
endlayer1 = 14;
endlayer2 = 17;

save_activations = true;

filenames = {'MIT_VGG_M_LR_average_448_jingzhao', 'MIT_VGG_M_LR_bilinear_448_final_1',...
    'MIT_VGG_M_LR_compact2_8192_fixW_448_final_1'};
epochs = [30, 42,  20];

for i = 2:numel(epochs)
    
    filename = filenames(i);
    epoch = epochs(i);
    
    netlink = strcat({'shallow_models/'}, filename, {'/net-epoch-'}, {num2str(epoch)}, {'.mat'});
    net = load(netlink{1});
    net = net.net;
    filename = filename{1};
    compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)
    
end


%% CUB VGGM bilinear average compactTS

dataset='CUB';
network='VGG_M';
batchSize = 32;
use448 = true;
endlayer1 = 14;
endlayer2 = 17;

save_activations = true;

filenames = {'CUB_VGG_M_LR_average_448_jingzhao', 'CUB_VGG_M_LR_bilinear_448_final_1',...
    'CUB_VGG_M_LR_compact2_8192_fixW_448_final_1'};
epochs = [30, 60,  60];

for i = 3
    
    filename = filenames(i);
    epoch = epochs(i);
    
    netlink = strcat({'shallow_models/'}, filename, {'/net-epoch-'}, {num2str(epoch)}, {'.mat'});
    net = load(netlink{1});
    net = net.net;
    filename = filename{1};
    compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)
    
end


%% MIT VGG16 bilinear average compactTS

dataset='MIT';
network='VGG_16';
batchSize = 8;
use448 = true;
endlayer1 = 30;
endlayer2 = 33;

save_activations = true;

filenames = {'MIT_VGG_16_LR_average_448_jingzhao', 'MIT_VGG_16_LR_bilinear_448_final_1_batch16',...
    'MIT_VGG_16_LR_compact2_8192_fixW_448_final_1.batch8'};
epochs = [30, 51,  20];

for i = 2:numel(epochs)
    
    filename = filenames(i);
    epoch = epochs(i);
    
    netlink = strcat({'shallow_models/'}, filename, {'/net-epoch-'}, {num2str(epoch)}, {'.mat'});
    net = load(netlink{1});
    net = net.net;
    filename = filename{1};
    compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)
    
end

%% CUB VGG16 average compactTS

dataset='CUB';
network='VGG_16';
batchSize = 8;
use448 = true;
endlayer1 = 30;
endlayer2 = 33;

save_activations = true;

filenames = {'CUB_VGG_16_LR_average_448_jingzhao', 'CUB_VGG_16_LR_compact2_8192_fixW_448_final_1', 'CUB_VGG_16_LR_bilinear_448_final_1'};
epochs = [30, 22, 13];

for i = 3
    
    filename = filenames(i);
    epoch = epochs(i);
    
    netlink = strcat({'shallow_models/'}, filename, {'/net-epoch-'}, {num2str(epoch)}, {'.mat'});
    net = load(netlink{1});
    net = net.net;
    filename = filename{1};
    compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)
    
end
%% MIT VGG16 finetuned compactTS

dataset='MIT';
network='VGG_16';
batchSize = 8;
use448 = false;
endlayer1 = 30;
endlayer2 = 33;

save_activations = true;


net = load('/mnt/x/yang/exp_data/fv_layer/exp_yang_fv/MIT_VGG_16_LR_fcfc_224_test_batch32/net-epoch-60.mat');
net = net.net;
net.layers = net.layers(1:30);
projDim = 8192;
nc = 512;
learnW = false;
TS_obj=yang_compact_bilinear_TS_nopool_2stream(projDim, [nc nc], 1);

wf=consts(dataset, 'projInitW', 'projDim', projDim, 'cfgId', 'compact2');
if exist(wf, 'file') ==2
    fprintf('loading an old compact2 weight\n');
    load(wf);
    % modify the initialized parameters to the loaded params
    TS_obj.h_={hs(1,:), hs(2,:)};
    TS_obj.weights_={ss(1,:), ss(2,:)};
    TS_obj.setSparseM(TS_obj.weights_, 1);
end
net.layers{end+1}=struct('type', 'custom',...
    'layerObj', TS_obj, ...
    'forward', @TS_obj.forward_simplenn, ...
    'backward', @TS_obj.backward_simplenn, ...
    'name', 'compact_TS', ...
    'outDim', projDim, ...
    'weights', {{cat(1, TS_obj.weights_{:})}}, ...
    'learningRate', [1]*learnW);
net.layers{end+1}=struct('type', 'custom',...
    'forward',  @yang_sqrt_forward, ...
    'backward', @yang_sqrt_backward, ...
    'name', 'sign_sqrt');
net.layers{end+1}=struct('type', 'custom',...
    'forward',  @yang_l2norm_forward, ...
    'backward', @yang_l2norm_backward, ...
    'name', 'L2_normalize');


filename = 'MIT_VGG16_FINETUNED_COMPACT_TS_448_8192';
compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)

%% MIT VGGM compact2 224

dataset='MIT';
network='VGG_M';
batchSize = 32;
use448 = false;
endlayer1 = 14;
endlayer2 = 17;

save_activations = true;

filename = 'MIT_VGG_M_LR_compact2_8192_fixW_224_final_1';
epoch = 60;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)

%% CUB VGGM compact2 224

dataset='CUB';
network='VGG_M';
batchSize = 32;
use448 = false;
endlayer1 = 14;
endlayer2 = 17;

save_activations = true;

filename = 'CUB_VGG_M_LR_compact2_8192_fixW_224_final_1';
epoch = 89;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)

%% CUB VGG16 compact2 224

dataset='CUB';
network='VGG_16';
batchSize = 32;
use448 = false;
endlayer1 = 30;
endlayer2 = 33;

save_activations = true;

filename = 'CUB_VGG_16_LR_compact2_8192_fixW_224_final_1';
epoch = 16;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)

%% MIT VGG16 compact2 224

dataset='MIT';
network='VGG_16';
batchSize = 32;
use448 = false;
endlayer1 = 30;
endlayer2 = 33;

save_activations = true;

filename = 'MIT_VGG_16_LR_compact2_8192_fixW_224_final_1';
epoch = 54;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)


%% CUB VGGM average 224

dataset='CUB';
network='VGG_M';
batchSize = 32;
use448 = false;
endlayer1 = 14;
endlayer2 = 17;

save_activations = true;

filename = 'CUB_VGG_M_LR_average_224_jingzhao';
epoch = 30;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;  

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)

%% MIT VGGM compact2 224

dataset='MIT';
network='VGG_M';
batchSize = 32;
use448 = false;
endlayer1 = 14;
endlayer2 = 17;

save_activations = true;

filename = 'MIT_VGG_M_LR_average_224_jingzhao';
epoch = 30;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)





%% CUB VGG16 average 224

dataset='CUB';
network='VGG_16';
batchSize = 8;
use448 = false;
endlayer1 = 30;
endlayer2 = 33;

save_activations = true;

filename = 'CUB_VGG_16_LR_average_224_jingzhao';
epoch = 19;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)

%% MIT VGG16 average 224

dataset='MIT';
network='VGG_16';
batchSize = 8;
use448 = false;
endlayer1 = 30;
endlayer2 = 33;

save_activations = true;

filename = 'MIT_VGG_16_LR_average_224_jingzhao';
epoch = 30;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)

%% CUB VGG16 fcfc 224 Replace with bilinear

dataset='CUB';
network='VGG_16';
batchSize = 8;
use448 = false;
endlayer1 = 30;
endlayer2 = 35;

save_activations = true;

filename = 'CUB_VGG_16_LR_fcfc_224_final_1';
epoch = 26;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_bilinear_fcfc(filename, dataset, network, endlayer1, use448, net, batchSize, save_activations, SVMopts)



%% MIT VGG16 fcfc 224 Replace with bilinear

dataset='MIT';
network='VGG_16';
batchSize = 8;
use448 = false;
endlayer1 = 30;
endlayer2 = 35;

save_activations = true;

filename = 'MIT_VGG_16_LR_fcfc_224_test_batch32';
epoch = 60;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_bilinear_fcfc(filename, dataset, network, endlayer1, use448, net, batchSize, save_activations, SVMopts)

%% MIT VGGM fcfc 224 Replace with bilinear

dataset='MIT';
network='VGG_M';
batchSize = 32;
use448 = false;
endlayer1 = 14;
endlayer2 = 19;

save_activations = true;

filename = 'MIT_VGG_M_LR_fcfc_224_final_1';
epoch = 20;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_bilinear_fcfc(filename, dataset, network, endlayer1, use448, net, batchSize, save_activations, SVMopts)

%% CUB VGGM fcfc 224 Replace with bilinear

dataset='CUB';
network='VGG_M';
batchSize = 32;
use448 = false;
endlayer1 = 14;
endlayer2 = 19;

save_activations = true;

filename = 'CUB_VGG_M_LR_fcfc_224_final_1';
epoch = 36;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_bilinear_fcfc(filename, dataset, network, endlayer1, use448, net, batchSize, save_activations, SVMopts)


%% CUB VGG16 average 224 Replace with bilinear

dataset='CUB';
network='VGG_16';
batchSize = 8;
use448 = false;
endlayer1 = 30;
endlayer2 = 33;

save_activations = true;

filename = 'CUB_VGG_16_LR_average_224_jingzhao';
epoch = 19;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_bilinear_fcfc(filename, dataset, network, endlayer1, use448, net, batchSize, save_activations, SVMopts)



%% MIT VGG16 average 224 Replace with bilinear

dataset='MIT';
network='VGG_16';
batchSize = 8;
use448 = false;
endlayer1 = 30;
endlayer2 = 33;

save_activations = true;

filename = 'MIT_VGG_16_LR_average_224_jingzhao';
epoch = 30;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_bilinear_fcfc(filename, dataset, network, endlayer1, use448, net, batchSize, save_activations, SVMopts)

%% MIT VGGM fcfc 224 Replace with bilinear

dataset='MIT';
network='VGG_M';
batchSize = 32;
use448 = false;
endlayer1 = 14;
endlayer2 = 17;

save_activations = true;

filename = 'MIT_VGG_M_LR_average_224_jingzhao';
epoch = 30;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_bilinear_fcfc(filename, dataset, network, endlayer1, use448, net, batchSize, save_activations, SVMopts)

%% CUB VGGM fcfc 224 Replace with bilinear

dataset='CUB';
network='VGG_M';
batchSize = 32;
use448 = false;
endlayer1 = 14;
endlayer2 = 17;

save_activations = true;

filename = 'CUB_VGG_M_LR_average_224_jingzhao';
epoch = 30;
netlink = ['shallow_models/' filename '/net-epoch-' num2str(epoch) '.mat'];
net = load(netlink);
net = net.net;

compare_bilinear_fcfc(filename, dataset, network, endlayer1, use448, net, batchSize, save_activations, SVMopts)





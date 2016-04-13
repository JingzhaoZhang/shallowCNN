%% load the pool5 features
featurePath=['/x/yang/exp_data/fv_layer/exp_yang_fv/',...
             'poolout_CUB_VGG_16_conv5_relu_448.mat'];
load(featurePath);
% load the imdb
partFileLoc='/x/yang/exp_data/fv_layer/exp01/cub-seed-01/imdb/parts.mat';
load(partFileLoc);
%% try simple average pooling
addpath ./no_publish/layers/pooling/
addpath ./no_publish/layers/normalize/

[trainFV_ave, valFV_ave]=pool_ave(trainFV, valFV); 
[trainFV_root, valFV_root]=normalize_root(trainFV_ave, valFV_ave);
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_root, valFV_root);
train_test_vlfeat('LR', trainFV_l2, trainY, valFV_l2, valY);
% accuracy 82.1%, mAP 83.1%
%% the above accuracy is suprisingly high, must have a sanity check
% a check of no normalizing
train_test_vlfeat('LR', trainFV_ave, trainY, valFV_ave, valY);
% accuracy 75.3%, mAP 77.6%

% check from FCFC finetuned network's activation
% TODO
%% classifier introspection.
[w, b, acc, map, cls]=...
    train_test_vlfeat('LR', trainFV_l2, trainY, valFV_l2, valY);
% w has size 512*200, b has size: 1*200
[~, prediction] = max(cls, [], 2);
%% simple visualization attempt
introspection_visualize(1028, w,b, valFV, valY, prediction, imdb);

%% conv5 visualization
visualize_conv5(24, valFV, valY, imdb, w);




%% try root -> ave -> l2 -> classification
[trainFV_root, valFV_root]=normalize_root(trainFV, valFV);
[trainFV_ave, valFV_ave]=pool_ave(trainFV_root, valFV_root); 
[trainFV_l2, valFV_l2]=normalize_l2(trainFV_ave, valFV_ave);
train_test_vlfeat('LR', trainFV_l2, trainY, valFV_l2, valY);
% accuracy is 77.0%, mAP is 79.3%
%% and instrospection
introspection_visualize(1, w,b, valFV_root, valY);
%% try ave -> root -> classification
[trainFV_ave, valFV_ave]=pool_ave(trainFV, valFV); 
[trainFV_root, valFV_root]=normalize_root(trainFV_ave, valFV_ave);
train_test_vlfeat('LR', trainFV_root, trainY, valFV_root, valY);
% accuracy 80.8%, mAP 81.0%
%% explore the classification on top of parts
% setup some varialable
partFileLoc='/x/yang/exp_data/fv_layer/exp01/cub-seed-01/imdb/parts.mat';
load(partFileLoc);
addpath no_publish/parts/
%% recall the structure of the saved feature
fpath=consts('CUB', 'poolout', 'network', 'VGG_16', ...
             'cfgId', 'compact2_finetuned', ...
             'use448', true, 'projDim', 8192);
load(fpath);         

% this accuracy is 82%, map=84.8%, a little less than the finetuned
train_test_vlfeat('LR', trainFV, trainY, valFV, valY);
%% 
%getIntermediateActivations
% then get the file:
% /home/yang/exp_data/fv_layer/exp_yang_fv/poolout_CUB_VGG_16_conv5_relu_448.mat
%% extract parts feature
[trainFV_part, valFV_part]=part_get_feature(trainFV, valFV, imdb);
outpath='/home/yang/exp_data/fv_layer/exp_yang_fv/poolout_CUB_VGG_16_conv5_relu_parts_448.mat';
savefast(outpath,'trainFV_part','trainY','valFV_part','valY');
%%
% a test on all parts, inlcuding the whole image one
% accuracy 77.8%, mAP 79.2%
train_test_vlfeat('LR', trainFV_part, trainY, valFV_part, valY);
%% try some classification, on all parts, only head, etc.
getInter=@(ibatch, batchSize, upper) ...
    ((ibatch-1)*batchSize+1): min(upper, ibatch*batchSize);
channel_selected=3;
trainFV_head=trainFV_part(getInter(channel_selected, 8192, +inf), :);
valFV_head=valFV_part(getInter(channel_selected, 8192, +inf), :);
train_test_vlfeat('LR', trainFV_head, trainY, valFV_head, valY);
% reproduce whole picture classification
% accuracy is 81.6%, mAP 84.5

% selected channel = 6, forehead
% accuracy is 27.7, mAP is 25.7

% selected channel = 3, belly
% accuracy is 34.6, mAP is 36.8
%% some visualizations of extracted parts


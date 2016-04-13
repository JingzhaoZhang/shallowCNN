function compare_shallow_regular(filename, dataset, network, endlayer1, endlayer2, use448, net, batchSize, save_activations, SVMopts)
%COMPARE_SHALLOW_REGULAR First enlayer ends with dense output. Second ends
%after compact + sqrt + l2.
% Method must be SVM or LR to start
activation_file1 = ['shallow_models/get_activation_saves/' filename '_layer_' num2str(endlayer1) '_activations.mat'];
activation_file2 = ['shallow_models/get_activation_saves/' filename '_layer_' num2str(endlayer2) '_activations.mat'];
% 
if strcmp(SVMopts.method1, 'SVM') ||strcmp(SVMopts.method1, 'LR')

if exist(activation_file1, 'file') == 2
    load(activation_file1)
else
    [trainFV, trainY, valFV, valY]=...
        get_activations_dataset_network_layer(...
        dataset, network, endlayer1, use448, net, batchSize);
    
    trainFV = sqrt(squeeze(mean(mean(trainFV, 1), 2)));
    norms = sqrt(sum(trainFV.^2, 1));
    trainFV = bsxfun(@rdivide, trainFV, norms);
    valFV = sqrt(squeeze(mean(mean(valFV, 1), 2)));
    norms = sqrt(sum(valFV.^2, 1));
    valFV = bsxfun(@rdivide, valFV, norms);
    if save_activations
        save(activation_file1, 'trainFV', 'valFV', 'trainY', 'valY');
        pause(10)
    end

end

classificationType = SVMopts.method1;
[w, b, acc, map, scores]= train_test_vlfeat(classificationType, SVMopts, ...
    squeeze(trainFV), squeeze(trainY), squeeze(valFV), squeeze(valY));
save(['shallow_models/' filename '_' classificationType '_shallow'],'acc','scores', 'w', 'b', 'map');
else
    display('Unknown method')
end

if strcmp(SVMopts.method2, 'SVM') ||strcmp(SVMopts.method2, 'LR')


if exist(activation_file2, 'file') == 2
    load(activation_file2)
else
    [trainFV, trainY, valFV, valY]=...
        get_activations_dataset_network_layer(...
        dataset, network, endlayer2, use448, net, batchSize);
    
    if ndims(trainFV) == 4
        trainFV = sqrt(squeeze(mean(mean(trainFV, 1), 2)));
        norms = sqrt(sum(trainFV.^2, 1));
        trainFV = bsxfun(@rdivide, trainFV, norms);
        valFV = sqrt(squeeze(mean(mean(valFV, 1), 2)));
        norms = sqrt(sum(valFV.^2, 1));
        valFV = bsxfun(@rdivide, valFV, norms);
    end
    
    if save_activations
        save(activation_file2, 'trainFV', 'valFV', 'trainY', 'valY', '-v7.3');
        pause(10)
    end
end



classificationType = SVMopts.method2;
[w, b, acc, map, scores]= train_test_vlfeat(classificationType, SVMopts,...
    squeeze(trainFV), squeeze(trainY), squeeze(valFV), squeeze(valY));
save(['shallow_models/' filename '_' classificationType '_regular'],'acc','scores', 'w', 'b', 'map');
display('Done!!!')
else
    display('Unknown method')
end
end


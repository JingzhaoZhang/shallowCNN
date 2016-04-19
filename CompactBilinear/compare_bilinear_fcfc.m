function compare_bilinear_fcfc(filename, dataset, network, endlayer1, use448, net, batchSize, save_activations, SVMopts)
%COMPARE_SHALLOW_REGULAR First enlayer ends with dense output. Second ends
%after compact + sqrt + l2.
% Method must be SVM or LR to start
activation_file1 = ['shallow_models/get_activation_saves/' filename '_layer_' num2str(endlayer1) 'compactTS_activations.mat'];

%



if strcmp(SVMopts.method1, 'SVM') ||strcmp(SVMopts.method1, 'LR')
    
    if exist(activation_file1, 'file') == 2
        load(activation_file1)
    else
        net.layers = net.layers(1:endlayer1);
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
        
        [trainFV, trainY, valFV, valY]=...
            get_activations_dataset_network_layer(...
            dataset, network, endlayer1+3, use448, net, batchSize);
        

        if save_activations
            save(activation_file1, 'trainFV', 'valFV', 'trainY', 'valY');
            pause(10)
        end
        
    end
    
    classificationType = SVMopts.method1;
    [w, b, acc, map, scores]= train_test_vlfeat(classificationType, SVMopts, ...
        squeeze(trainFV), squeeze(trainY), squeeze(valFV), squeeze(valY));
    save(['shallow_models/' filename '_' classificationType '_ReplaceWithcompactTS'],'acc','scores', 'w', 'b', 'map');
else
    display('Unknown method')
end


end


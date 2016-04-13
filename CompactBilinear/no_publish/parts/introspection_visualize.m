function introspection_visualize(id, w, b, conv5, y, pre, imdb)
    % deal with imdb part, show original image
    index=find(imdb.images.set == 3);
    raw_index = index(id);
    imagePath=fullfile(imdb.imageDir, imdb.images.name{raw_index});
    subplot(2, 2, 1);
    imshow(imagePath)
    title('original image')
    
    % prepare some short hands
    [nh, nw, nc, nn] = size(conv5);
    feature=squeeze(conv5(:,:,:,id));
    label=y(id);
    sign_sqrt=@(x) sqrt(abs(x)).* sign(x);
    % not using l2norm
    % adding l2norm after sign_sqrt makes the plot wierd. 
    l2norm=@(x) x/sqrt(sum(x(:).^2));
    
    % TODO(We could also visualize the second largest activation)
    weights_id=[label 1 43];
    for i=1:numel(weights_id)
        wi=weights_id(i);
        % get out the weight
        weight= reshape(w(:, wi), 1, 1, []);
        activations=dot(sign_sqrt(feature),...
                        repmat(weight, nh, nw, 1), 3)+b(wi);
        subplot(2,2,i+1);
        imagesc(activations);
        colorbar
        
        title(['Actual=', num2str(label), ...
               ', predicted=', num2str(pre(id)), ...
               ', weights=', num2str(wi)]);
    end
    
end
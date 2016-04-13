function visualize_conv5(id, conv5, y, imdb, w)
    
    % prepare some short hands
    [nh, nw, nc, nn] = size(conv5);
    feature=squeeze(conv5(:,:,:,id));
    label=y(id);
  
    % figure out how many channels to show
    threshold=15;
    max_act=max(reshape(feature, [], size(feature, 3)), ...
                [], 1);
    ch_ind=max_act>threshold;
    msize=ceil(sqrt(sum(ch_ind)+1));
    
    % show original image
    index=find(imdb.images.set == 3);
    raw_index = index(id);
    % reading image
    getbatch=getBatchWrapper(struct('numThreads', 4,...
          'imageSize', [448 448 3], 'averageImage', [0 0 0]));
    a=getbatch(imdb, raw_index);
    a=uint8(a);
    % display
    subplot(msize, msize, 1);
    imshow(a)
    title('original image')
    
    % showing the activations
    ch_index=find(ch_ind);
    for i=1:numel(ch_index)
        subplot(msize, msize, i+1);
        ch=ch_index(i);
        imagesc(squeeze(feature(:,:,ch)));
        colorbar
        title(['channel ', num2str(ch)]);
    end
    
    % output the classifier weights of the selected
    fprintf('class %d, classifier weights: ', label);
    w(ch_index, label)
end
%% 
% this file extract part locations and imdb structure. Do classification in
% the parts_classify.m file. 

partFileLoc='/x/yang/exp_data/fv_layer/exp01/cub-seed-01/imdb/parts.mat';
addpath no_publish/parts/
%% read the part file
file_part='./data/cub/parts/part_locs.txt';
file_partnames='./data/cub/parts/parts.txt';
[imageid, partid, x, y, vis]=textread(file_part, '%d %d %f %f %d');
[~, strnames]=textread(file_partnames, '%d %s', 'whitespace', '\n');

% the structure of parts: ([x, y, isVis], partid, imageId)
parts=zeros(3, 15, 11788);
for i=1:length(imageid)
    parts(:, partid(i), imageid(i))=[x(i), y(i), vis(i)];
end

%%
imdb=load(consts('CUB', 'imdb'));

% read through image sizes and transform parts size to reshaped size
hw=zeros(2, 11788);
for i=1:length(imdb.images.name)
    iname=[imdb.imageDir '/' imdb.images.name{i}];
    tmp=imfinfo(iname);
    hw(:, i)=[tmp.Width, tmp.Height];
end

imdb.images.parts=parts;
imdb.images.hw=hw;
save(partFileLoc, 'imdb');
%% 
load(partFileLoc);
% visualize some image
id=3; % input, also the imdb
imname=[imdb.imageDir '/' imdb.images.name{id}];
part_showKeyPoints(imname, imdb.images.parts(:,:,id));

%% convert part location in terms of the reshaped image
%   :explore with the getbatch reshape
getbatch=getBatchWrapper(struct('numThreads', 4,...
                        'imageSize', [448 448 3], 'averageImage', [0 0 0]));
id=3;
a=getbatch(imdb, id);
a=uint8(a);

part_showKeyPoints(a, ...
  part_original2reshape(imdb.images.parts(:,:,id), imdb.images.hw(:,id)));

%% simplify the getbatch reshape
b=imread(imname);
factor=max((448)/size(b,1), (448)/size(b,2));
b=imresize(b, factor);
start=size(b);
start=ceil((start(1:2)+1-448)/2);
b=b(start(1):(start(1)+448-1), start(2):(start(2)+448-1), :);
imshow(b)

%% have a map from original axis to new axis
% part_original2reshape
% and apply to the imdb structure
parts_transformed=zeros(3, 15, 11788);
for i=1: length(imdb.images.id)
    parts_transformed(:,:,i)= part_original2reshape(...
            imdb.images.parts(:,:,i), imdb.images.hw(:,i));
end
imdb.images.parts_transformed=parts_transformed;

save(partFileLoc, 'imdb');
%% have a map from reshaped image to feature map
% look at fast RCNN
% decided to directly rescale to the target size

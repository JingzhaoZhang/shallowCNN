% crop out the ground truth bounding box
% load the imdb
partFileLoc='/x/yang/exp_data/fv_layer/exp01/cub-seed-01/imdb/parts.mat';
load(partFileLoc);
%% ground truth box based on parts, abandoned. 
gid=4;
doVis=1;

imname=[imdb.imageDir '/' imdb.images.name{gid}];
imo=imread(imname);
parts=imdb.images.parts(:,:,gid);

% initialize tl and br
tl=size(imo); tl=tl(1:2);
br=[1 1];
for ip=1:15
    if parts(3, ip)>0
        tl=min(tl, parts(1:2, ip)');
        br=max(br, parts(1:2, ip)');
    end
end
if doVis
    subplot(1,2,1);
    part_showKeyPoints(imo, parts);

    subplot(1,2,2);
    imshow(imo);
    hold on
    plot(tl(1), tl(2), 'r*', 'MarkerSize', 10);
    plot(br(1), br(2), 'b+', 'MarkerSize', 10);
    tl
    br
end
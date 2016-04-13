nclass=20;

[idx, C, sumd]=kmeans(w, nclass, 'Distance','cityblock');
hist(idx)
figure

wp=w;

start=1;
for i=1:nclass
    now=w(idx==i, :);
    num=size(now, 1);
    
    wp(start: (start+num-1), :) = now;
    start=start+num;
end

imagesc(wp);
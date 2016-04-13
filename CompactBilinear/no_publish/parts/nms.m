function out=nms(im)
%% this appears to be super noisy
    [m, n] = size(im);
    out=true(m, n);
    % 8 way non maximum supression
    for i=-1:1:1
        for j=-1:1:1
            % the (i, j) element relative to the element now to be compared
            % add a safe region of btlr 1 to the output
            imij=zeros(m+2, n+2, 'like', im);
            imij(2:(m+1), 2:(n+1)) = im;
            % cut the imij
            imij = imij((2:(m+1))+i, (2:(n+1))+j);
            
            out = out & (im >=imij);
        end
    end
end
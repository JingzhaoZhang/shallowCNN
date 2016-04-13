function out=conv5_to_compact_TS(conv5, projDim)
    nbatch=100;
    [h,w,c,n]=size(conv5);
    out=zeros([h, w, projDim, n], 'single');
    obj=yang_compact_bilinear_TS_nopool_2stream(projDim, [c c], 0);
    
    for iout=1:nbatch:n
        fprintf('extracting compact features: id = %d\n',iout);
        
        inter=iout:min(n, iout+nbatch-1);
        tt=gpuArray(conv5(:,:,:,inter));
        projected=obj.forward({tt, tt}, []);
        projected=gather(projected{1});
        
        out(:,:,:, inter)=projected;
    end % end of for nbatch loop
    
end
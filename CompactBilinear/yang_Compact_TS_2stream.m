classdef yang_Compact_TS_2stream < dagnn.Pooling
    properties
        % NOTE!! and TODO!!: the parameter learning is not implemented yet
        % Because learning it doesn't lead to significant improvement.
        
        % these two should be set
        outDim = 4096  
        learnW = 0
        previousChannels = [256, 4096]

        % these are automatically set
        h_ = {}
        weights_ = {}
        % second method to speed up
        sparseM={}
        
        % batch size when doing the compact transform
        % mainly for saving memory when doing FFT
        bsn=8
    end

    methods
        
    function this=yang_Compact_TS_2stream(varargin)
        this.load(varargin) ;
%         assert(learnW==0, 'Do not support learning W yet');
%         
%         this.outDim=projDim;
%         this.learnW=learnW;
        % fix a random seed
        % such that we could reproduce without saving the random weights. 
        rng(1);


        this.h_={randi(this.outDim, 1, this.previousChannels(1)), ...
                 randi(this.outDim, 1, this.previousChannels(2))}; %hs
        this.weights_={randi(2, 1, this.previousChannels(1))*2-3, ...
                       randi(2, 1, this.previousChannels(2))*2-3}; %ss

        for i=1:2
            M=sparse(this.outDim, this.previousChannels(i));
            for j=1:this.previousChannels(i)
                M(this.h_{i}(j), j)=this.weights_{i}(j);
            end
            % should be removed if run on cpu
            this.sparseM{i}=M;
        end
    end
      
    
    function outputs = forward(self, inputs, params)
        if ~isa(self.sparseM{1}, 'gpuArray')
            for i=1:2
                self.sparseM{i}=gpuArray(self.sparseM{i});
            end
        end
        
        x=inputs{1}; 
        [h,w,c,n]=size(x);
        x=permute(x, [3,1,2,4]); % order c h w n
        x=reshape(x, c, h*w*n);
        
        % another input
        y=inputs{2};
        [hy,wy,cy,ny]=size(y);
        assert((h==hy) && (w==wy) && (n==ny), ...
        'Assertion Failed: Compact_TS_2stream forward, two inputs size different');
        y=permute(y, [3,1,2,4]); % order c h w n
        y=reshape(y, cy, h*w*n);
        
        out=ones([self.outDim,h,w,n], 'like', x);

        for img=1:ceil(n/self.bsn)
            interLarge=getInter(img, self.bsn*h*w, n*h*w);
            interSmall=getInter(img, self.bsn, n);
            
            ttt=forward_aBatch({x(:, interLarge), y(:, interLarge)}, self.sparseM);
            out(:, :,:, interSmall)=...
                reshape(ttt, self.outDim, h, w, numel(interSmall));
        end

        outputs{1}=permute(out, [2,3,1,4]);
    end
    

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
        ch=self.h_; % size: 2*c; range: 1~d
        cs=self.weights_; % size; 2*c; range: -1 ~ +1

        x=inputs{1};
        [h,w,c,n]=size(x);
        x=permute(x, [3,1,2,4]); % order c h w n
        x=reshape(x, c, h*w*n);
        
        % another input
        y=inputs{2};
        [hy,wy,cy,ny]=size(y);
        assert((h==hy) && (w==wy) && (n==ny), ...
        'Assertion Failed: Compact_TS_2stream backward, two inputs size different');
        y=permute(y, [3,1,2,4]); % order c h w n
        y=reshape(y, cy, h*w*n);
        
        cc=[c cy];

        out={zeros(cc(1), h*w*n, 'like', x), ...
             zeros(cc(2), h*w*n, 'like', x)};
        %dzdw=zeros(c, 1, 'single'); % should change to like
        %dzdw={dzdw, dzdw};

        now_dzdx = derOutputs{1};
        for img=1:ceil(n/self.bsn)
            interLarge=getInter(img, self.bsn*h*w, n*h*w);
            interSmall=getInter(img, self.bsn, n);

            batch_dzdx=now_dzdx(:,:,:,interSmall);
            batch_dzdx=permute(batch_dzdx, [3,1,2,4]);
            batch_dzdx=reshape(batch_dzdx, self.outDim, []);
            batch_dzdx=fft(batch_dzdx,[], 1);

            [out_tmp, t_dzdw]=backward_aBatch(batch_dzdx, ch, cs, self.sparseM,...
                {x(:, interLarge), y(:, interLarge)}, self.learnW);
            
            out{1}(:,interLarge)=out_tmp{1};
            out{2}(:,interLarge)=out_tmp{2};
            
            if self.learnW
                dzdw{1}=t_dzdw{1}+dzdw{1};
                dzdw{2}=t_dzdw{2}+dzdw{2};
            end
        end

        for i=1:2
            out{i}=reshape(out{i}, cc(i), h*w, n);
            out{i}=permute(out{i}, [2,1,3]); % order hw, c, n
            out{i}=reshape(out{i}, h,w,cc(i),n);
        end
        derInputs=out;
        derParams = {};
        if self.learnW
            derParams{1}={cat(2, dzdw{:})'};
        end
    end


    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1};
      outputSizes{1}(3) = self.outDim;
    end
    
    end
  
    methods (Static)
    %%% some helper functions for debug
    function output=bilinear(x, y)
        ind=size(x)==size(y);
        assert(all([ind(1) ind(2) ind(4)]));
        [h, w, c, n]=size(x);
        output=zeros([h, w, c*size(y,3), n], 'like', x);
        for ih=1:h
            for iw=1:w
                for in=1:n
                    xt=squeeze(x(ih, iw, :, in));
                    yt=squeeze(y(ih, iw, :, in));
                    t=xt*yt';
                    output(ih, iw, :, in)=t(:);
                end
            end
        end
    end
    
    function output = pairwiseInner(x)
        [h, w, c, n]=size(x);
        x=permute(x, [3, 1, 2, 4]);
        x=reshape(x, c, h*w*n);
        output=x'*x;
    end
   
    function compare2vec(x, y)
        diff=abs(x(:) - y(:));
        fprintf('max error=%f, average error=%f \n', max(diff), mean(diff));
        
        rel_diff=diff./ (abs(x(:)) + 0.0001);
        fprintf('rel_max error=%f, rel_average error=%f \n', ...
            max(rel_diff), mean(rel_diff));
    end
    
    end

end


%%%%%%%%%%%%%%%%%%%%% some helper functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out=forward_aBatch(x, sparseM)
    out=ones([size(sparseM{1},1), size(x{1}, 2)], 'like', x{1});
    
    for ipoly=1:2
        count=single(sparseM{ipoly}*double(x{ipoly}));
        
        count=fft(count, [], 1);
        out=out .* count;
    end
    
    out=real(ifft(out, [], 1));
end

function [out, dzdw]=backward_aBatch(repFftDzdy, ch, cs, sparseM, x, learnW)
    dzdw=cell(1,2);
    out=cell(1,2);
    
    for ipoly=1:2
        count=single(sparseM{ipoly}*double(x{ipoly}));
        
        hip=ch{3-ipoly};
        sip=cs{3-ipoly};        
        count(2:end, :)=flipud(count(2:end, :));

        count=fft(count, [], 1);
        count=count .* repFftDzdy;
        dLdq=real(ifft(count ,[] ,1));
        
        out{3-ipoly}=bsxfun(@times, dLdq(hip,:), sip');
        if learnW
            dzdw{3-ipoly}=sum(dLdq(hip, :).*x{ipoly},2);
        end
    end
end

function out=getInter(iseg, segLen, upper)
    out=gpuArray( ((iseg-1)*segLen+1) : min(upper, iseg*segLen) );
end


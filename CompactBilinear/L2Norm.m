classdef L2Norm < dagnn.Pooling
  properties
      dimension = 1
      epsilon = 0.1
      scale = 1
      % dim 1 : normalize each window across h and w.
      % dim 0 : normalize the entire datablock.
      % dim 2 : Normalize each pixel accross channel.
  end

  methods
    function outputs = forward(self, inputs, params)
% 	display('L2Norm');
%     display(size(inputs{1}));
    X=inputs{1};
    if self.dimension==1
        norms = sqrt(sum(sum(X.^2, 1), 2));
        X=bsxfun(@rdivide, X, (norms+self.epsilon));
    elseif self.dimension==0
       norms = sqrt(sum(sum(sum(X.^2, 1), 2),3));
       X=bsxfun(@rdivide, X, (norms+self.epsilon));
    elseif self.dimension==2
       norms = sqrt(sum(X.^2,3));
       X=bsxfun(@rdivide, X, (norms+self.epsilon));
    end
    outputs{1} = X*self.scale;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
	dzdy=derOutputs{1};
	X=inputs{1};
    if self.dimension==1
        lambda=1./(sqrt(sum(sum(X.^2, 1),2)) + self.epsilon);
        dzdx=bsxfun(@times, lambda, dzdy) - bsxfun(@times, X, (lambda.^3) .* sum(sum(X.*dzdy, 1),2));     
    elseif self.dimension==0
        lambda=1./(sqrt(sum(sum(sum(X.^2, 1),2), 3)) + self.epsilon);
        dzdx=bsxfun(@times, lambda, dzdy) - bsxfun(@times, X, (lambda.^3) .* sum(sum(sum(X.*dzdy, 1),2),3));
    elseif self.dimension ==2
        lambda=1./(sqrt(sum(X.^2, 3)) + self.epsilon);
        dzdx=bsxfun(@times, lambda, dzdy) - bsxfun(@times, X, (lambda.^3) .* sum(X.*dzdy,3));

    end    
    
    derInputs{1}=dzdx*self.scale;
    derParams={};
    end



    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1};
    end

    function obj = L2Norm(varargin)
      obj.load(varargin) ;
    end
  end




end


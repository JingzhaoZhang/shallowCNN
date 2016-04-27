classdef Sqrt < dagnn.Pooling
  properties
  end

  methods
    function outputs = forward(self, inputs, params)
%         display('sqrt');
%         display(size(inputs{1}));
    in=inputs{1};
    outputs{1} =sign(in).*sqrt(abs(in));
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      ep=1e-1;
      derInputs{1}=derOutputs{1} .* 0.5 ./ (sqrt(abs(inputs{1})) + ep);
      derParams = {} ;
    end



    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1};
    end

    function obj = Sqrt(varargin)
      obj.load(varargin) ;
    end
  end


end


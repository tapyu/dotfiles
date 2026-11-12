classdef BiasLayer 
    %nnet.onnx.layer.BiasLayer   A class to load a
    % nnet.onnx.layer.BiasLayer from a previous release and convert it
    % to a nnet.onnx.layer.ElementwiseAffineLayer.
    %
    %   Copyright 2018 The MathWorks, Inc.
    
    methods(Static)
        function obj = loadobj(s)
            assert(isfield(s, 'Bias'));
            obj = nnet.onnx.layer.ElementwiseAffineLayer(s.Name, 1, s.Bias);
        end
    end
end
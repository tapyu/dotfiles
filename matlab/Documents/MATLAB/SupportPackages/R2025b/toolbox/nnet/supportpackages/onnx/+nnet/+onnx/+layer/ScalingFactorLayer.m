classdef ScalingFactorLayer
    %nnet.onnx.layer.ScalingFactorLayer   A class to load a
    % nnet.onnx.layer.ScalingFactorLayer from a previous release and
    % convert it to a nnet.onnx.layer.ElementwiseAffineLayer.
    %
    %   Copyright 2018 The MathWorks, Inc.
    
    methods(Static)
        function obj = loadobj(s)
            assert(isfield(s, 'Scale'));
            obj = nnet.onnx.layer.ElementwiseAffineLayer(s.Name, s.Scale, 0);
        end
    end
end
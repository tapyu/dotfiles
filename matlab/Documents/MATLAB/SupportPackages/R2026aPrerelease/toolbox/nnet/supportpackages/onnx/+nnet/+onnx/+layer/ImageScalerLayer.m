classdef ImageScalerLayer
    %nnet.onnx.layer.ImageScalerLayer   A class to load a
    % nnet.onnx.layer.ImageScalerLayer from a previous release and convert
    % it to a nnet.onnx.layer.ElementwiseAffineLayer.
    %
    %   Copyright 2018 The MathWorks, Inc.
    
    methods(Static)
        function obj = loadobj(s)
            assert(isfield(s, 'Scale') && isfield(s, 'Bias'));
            % Bias : list of floats. Bias applied to each channel, same size as C.
            % Scale : float (scalar)
            obj = nnet.onnx.layer.ElementwiseAffineLayer(s.Name, s.Scale, reshape(s.Bias, [1 1 numel(s.Bias)]));
        end
    end
end
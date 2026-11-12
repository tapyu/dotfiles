classdef PlaceholderOutputLayer < nnet.cnn.layer.PlaceholderLayer
%PlaceholderOutputLayer   A layer to take the place of an unknown output layer
%   See also importONNXLayers.
% 

%   Copyright 2019-2021 The MathWorks, Inc.
    methods
        function this = PlaceholderOutputLayer(name)
            this@nnet.cnn.layer.PlaceholderLayer(name);
            this.Description = getString(message('nnet_cnn_onnx:onnx:OutputPlaceholderDescription'));
            this.Type = getString(message('nnet_cnn_onnx:onnx:PlaceholderType'));
        end
    end
end

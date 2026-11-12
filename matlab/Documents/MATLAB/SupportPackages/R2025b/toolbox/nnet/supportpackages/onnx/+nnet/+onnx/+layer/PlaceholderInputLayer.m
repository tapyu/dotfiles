classdef PlaceholderInputLayer < nnet.cnn.layer.PlaceholderLayer
%PlaceholderInputLayer   A layer to take the place of an unknown input layer
%   See also importONNXLayers.
% 

%   Copyright 2021 The MathWorks, Inc.
    methods
        function this = PlaceholderInputLayer(name)
            this@nnet.cnn.layer.PlaceholderLayer(name);
            this.Description = getString(message('nnet_cnn_onnx:onnx:InputPlaceholderDescription'));
            this.Type = getString(message('nnet_cnn_onnx:onnx:PlaceholderType'));
        end
    end
end

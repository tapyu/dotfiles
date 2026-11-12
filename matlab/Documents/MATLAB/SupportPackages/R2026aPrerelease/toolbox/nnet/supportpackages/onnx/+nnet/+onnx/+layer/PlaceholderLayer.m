classdef PlaceholderLayer < nnet.cnn.layer.PlaceholderLayer
%PlaceholderLayer   A layer to take the place of an unsupported ONNX layer
% 
%   PlaceholderLayer properties:
%      ONNXNode	   - A struct containing the ONNX NodeProto for this layer.
%      Weights - A struct containing weights imported from the ONNX for this layer, if any.
% 
%   See also importONNXLayers.
% 
%   Copyright 2017-2019 The MathWorks, Inc.
    
    properties
        ONNXNode     % A struct containing the ONNX configuration for this layer.
        Weights = [];
    end
    
    methods
        function this = PlaceholderLayer(name, node, numInputs, numOutputs)
            assert(numInputs > 0);
            assert(numOutputs > 0);
            this@nnet.cnn.layer.PlaceholderLayer(name);
            this.ONNXNode = node;
            this.Description = getString(message('nnet_cnn_onnx:onnx:PlaceholderDescription', node.op_type));
            this.Type = getString(message('nnet_cnn_onnx:onnx:PlaceholderType'));
            this.NumInputs = numInputs;
            this.NumOutputs = numOutputs;
        end
        
        function varargout = predict( ~, varargin )
            throwAsCaller(MException(message('nnet_cnn_onnx:onnx:PlaceholderCantBeUsed')));
        end
        
        function varargout = forward( ~, varargin )
            throwAsCaller(MException(message('nnet_cnn_onnx:onnx:PlaceholderCantBeUsed')));
        end
        
        function varargout = backward( ~, varargin )
            throwAsCaller(MException(message('nnet_cnn_onnx:onnx:PlaceholderCantBeUsed')));
        end
    end
end

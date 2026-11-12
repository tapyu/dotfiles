classdef ConnectComponentsLayer < nnet.layer.Layer & nnet.layer.Formattable & nnet.internal.cnn.layer.Traceable
    % A MIMO layer that passes through inputs unchanged. Used to connect
    % disconnected components into a single connected component, which is
    % required in DLT networks.
    
    %   Copyright 2021 The MathWorks, Inc.
    methods
        function this = ConnectComponentsLayer(name, numInputs)
            assert(isstring(name) || ischar(name), ...
                message('nnet_cnn_onnx:onnx:LayerNameArg'));
            this.Name = name;
            this.Description = getString(message('nnet_cnn_onnx:onnx:ConnectComponentsLayerDescription'));
            this.NumInputs = numInputs;
            this.NumOutputs = numInputs;
        end
        
        function varargout = predict( this, varargin )
            assert(numel(varargin)==this.NumInputs, message('nnet_cnn_onnx:onnx:ConnectComponentsLayerWrongNumInputs'));
            varargout = varargin;
        end
    end
end

classdef LogLayer < nnet.layer.Layer ...
    & nnet.internal.cnn.layer.Traceable ...
    & nnet.layer.Elementwise
    
% LogLayer   Log layer
%
%   layer = LogLayer(Name) creates a log unit layer with
%   name Name. This type of layer calculates Y = log(X).

%   Copyright 2021-2024 The MathWorks, Inc.
%#codegen
    methods
        function this = LogLayer(Name)
            this.Name = Name;
            if coder.target('MATLAB')                
                this.Description = getString(message('nnet_cnn_onnx:onnx:LogDescription'));
                this.Type = getString(message('nnet_cnn_onnx:onnx:LogType'));
            end
        end
        
        function Z = predict(~, X)
            Z = log(X);
        end
        
        function dLdX = backward(~, X, ~, dLdZ, ~)
            dLdX = dLdZ./X;
        end
    end
end


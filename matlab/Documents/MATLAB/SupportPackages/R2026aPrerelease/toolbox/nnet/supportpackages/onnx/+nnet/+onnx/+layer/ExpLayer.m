classdef ExpLayer < nnet.layer.Layer ...
    & nnet.internal.cnn.layer.Traceable ...
    & nnet.layer.Elementwise

% ExpLayer   Exp layer
%
%   layer = ExpLayer(Name) creates a log unit layer with
%   name Name. This type of layer calculates Y = exp(X).

%   Copyright 2021-2024 The MathWorks, Inc.
%#codegen
    methods
        function this = ExpLayer(Name)
            this.Name = Name;
            if coder.target('MATLAB')
                this.Description = getString(message('nnet_cnn_onnx:onnx:ExpDescription'));
                this.Type = getString(message('nnet_cnn_onnx:onnx:ExpType'));
            end
        end
        
        function Z = predict(~, X)
            Z = exp(X);
        end
        
        function dLdX = backward(~, ~, Z, dLdZ, ~)
            dLdX = dLdZ.*Z;
        end
    end
end


classdef IdentityLayer < nnet.layer.Layer ...
    & nnet.internal.cnn.layer.Traceable
    % Layer implementing the ONNX Identity operator
    
    %   Copyright 2018-2020 The MathWorks, Inc.
    methods
        function this = IdentityLayer(name)
            assert(isstring(name) || ischar(name), ...
                message('nnet_cnn_onnx:onnx:LayerNameArg'));
            this.Name = name;
        end
        
        function Z = predict( this, X )
            Z = X;
        end
        
        function dLdX = backward( this, X, Z, dLdZ, memory )
            dLdX = dLdZ;
        end
    end
end

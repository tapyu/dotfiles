classdef ElementwiseAffineLayer < nnet.layer.Layer ...
    & nnet.internal.cnn.layer.Traceable
    %ElementwiseAffineLayer
    %
    %   layer = ElementwiseAffineLayer(Name, Scale, Offset) creates a layer
    %   with name Name that applies an elementwise scaling followed by an
    %   addition to the input. Scale and Offset can be any shape that is
    %   expandable to the input tensor shape.
    
    %   Copyright 2018-2021 The MathWorks, Inc.
    properties
        Scale
        Offset
        DoScale
        DoOffset
    end
    
    methods
        function this = ElementwiseAffineLayer(name, Scale, Offset)
            if ~(isstring(name) || ischar(name))
                error(message('nnet_cnn_onnx:onnx:ElementwiseAffineArg1'));
            end
            if isempty(Scale) || ~(isnumeric(Scale) && isreal(Scale))
                error(message('nnet_cnn_onnx:onnx:ElementwiseAffineArg2'));
            end
            if isempty(Offset) || ~(isnumeric(Offset) && isreal(Offset))
                error(message('nnet_cnn_onnx:onnx:ElementwiseAffineArg2'));
            end
            this.Name        = name;
            this.Description = getString(message('nnet_cnn_onnx:onnx:ElementwiseAffineDescription'));
            this.Type        = getString(message('nnet_cnn_onnx:onnx:ElementwiseAffineType'));
            this.Scale       = Scale;
            this.Offset      = Offset;
            this.DoScale     = any(Scale(:) ~= 1);
            this.DoOffset    = any(Offset(:) ~= 0);
        end
        
        function Z = predict( this, X )
            Z = X;
            
            %Typecasting to workaround issue caused by dlnetwork datatype
            if this.DoScale
                Z = Z.*cast(this.Scale,'like',X);
            end
            if this.DoOffset
                Z = Z + cast(this.Offset,'like',X);
            end
        end
        
        function dLdX = backward( this, X, Z, dLdZ, memory )
            dLdX = dLdZ;
            if this.DoScale
                dLdX = dLdX.*cast(this.Scale,'like',dLdX);
            end
        end
    end
end
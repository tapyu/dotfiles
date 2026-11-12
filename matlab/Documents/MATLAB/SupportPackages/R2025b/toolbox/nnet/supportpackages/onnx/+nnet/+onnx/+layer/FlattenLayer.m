classdef FlattenLayer < nnet.layer.Layer ...
    & nnet.internal.cnn.layer.Traceable
    % FlattenLayer   Flatten layer ONNX-style
    %
    %   layer = FlattenLayer(Name) creates a layer with name Name that
    %   flattens the MATLAB input tensor in the way ONNX does. The ONNX
    %   layer has input shape [N C H W] and output shape [N CHW], where the
    %   [C H W] dimensions have been flattened in that order in C-style.
    %   The corresponding MATLAB layer has input shape [H W C N] and will
    %   need output shape [1 1 CHW N], where [C H W] is flattened in that
    %   order and C-style, as in ONNX. [C H W] C-style has the same linear
    %   ordering as [W H C] Fortran-style. So all we need to do is create
    %   [W H C N] in MATLAB and we will have the desired linear ordering
    %   for each N. Then we just reshape to [1 1 CHW N]. In the backward
    %   pass, we reshape the gradient back to [W H C N] then permute to the
    %   shape of X, [H W C N].
    
    %   Copyright 2018-2020 The MathWorks, Inc.
    methods
        function this = FlattenLayer(name)
            this.Name = name;
            this.Description = getString(message('nnet_cnn_onnx:onnx:FlattenDescription'));
            this.Type = getString(message('nnet_cnn_onnx:onnx:FlattenType'));
        end
        
        function Z = predict( this, X )
            % X is size [H W C N].
            % Z is size [1 1 HWC N].
            [h,w,c,n] = size(X);
            Z = reshape(permute(X,[2 1 3 4]), [1 1 h*w*c n]);
        end
        
        function dLdX = backward( this, X, Z, dLdZ, memory )
            % dLdZ is size [1 1 HWC N].
            % dLdX and X are size [H W C N].
            [h,w,c,n] = size(X);
            dLdX = permute(reshape(dLdZ, [w h c n]), [2 1 3 4]);
        end
    end
end

classdef Flatten3dLayer < nnet.layer.Layer ...
    & nnet.internal.cnn.layer.Traceable
    % Flatten3dLayer   Flatten layer ONNX-style
    %
    %   layer = Flatten3dLayer(Name) creates a layer with name Name that
    %   flattens the MATLAB input tensor in the way ONNX does. The ONNX
    %   layer has input shape [N C H W D] and output shape [N CHWD], where the
    %   [C H W D] dimensions have been flattened in that order in C-style.
    %   The corresponding MATLAB layer has input shape [H W D C N] and will
    %   need output shape [1 1 1 CHWD N], where [C H W D] is flattened in that
    %   order and C-style, as in ONNX. [C H W D] C-style has the same linear
    %   ordering as [D W H C] Fortran-style. So all we need to do is create
    %   [D W H C N] in MATLAB and we will have the desired linear ordering
    %   for each N. Then we just reshape to [1 1 1 CHWD N]. In the backward
    %   pass, we reshape the gradient back to [D W H C N] then permute to the
    %   shape of X, [H W D C N].
    
    %   Copyright 2019-2020 The MathWorks, Inc.
    methods
        function this = Flatten3dLayer(name)
            this.Name = name;
            this.Description = getString(message('nnet_cnn_onnx:onnx:FlattenDescription'));
            this.Type = getString(message('nnet_cnn_onnx:onnx:FlattenType'));
        end
        
        function Z = predict( this, X )
            % X is size [H W D C N].
            % Z is size [1 1 1 HWCD N].
            [h,w,d,c,n] = size(X);
            Z = reshape(permute(X,[3 2 1 4 5]), [1 1 1 h*w*d*c n]);
        end
        
        function dLdX = backward( this, X, Z, dLdZ, memory )
            % dLdZ is size [1 1 1 HWDC N].
            % dLdX and X are size [H W D C N].
            [h,w,d,c,n] = size(X);
            dLdX = permute(reshape(dLdZ, [d w h c n]), [3 2 1 4 5]);
        end
    end
end

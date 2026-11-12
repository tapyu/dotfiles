classdef Flatten3dInto2dLayer < nnet.layer.Layer & nnet.layer.Formattable ...
        & nnet.internal.cnn.layer.Traceable
    % Flatten3dInto2dLayer   Flatten layer ONNX-style into a 2D array
    %
    %   layer = Flatten3dInto2dLayer(Name) creates a layer that flattens a
    %   MATLAB 3D image batch in the way ONNX does, producing a 2D output
    %   array with CB format. The ONNX layer has input shape [N C H W D]
    %   and output shape [N CHWD], where the [C H W D] dimensions have been
    %   flattened in that order in C-style. The corresponding MATLAB layer
    %   has input shape [H W D C N] and will have an output shape of [CHWD
    %   N]. [C H W D] C-style has the same linear ordering as [D W H C]
    %   Fortran-style. So this layer creates [D W H C N] in MATLAB, which
    %   has the desired linear ordering for each N. Then it reshapes to
    %   [CHWD N].

    %   Copyright 2021 The MathWorks, Inc.

    methods
        function this = Flatten3dInto2dLayer(name)
            this.Name = name;
            this.Description = getString(message('nnet_cnn_onnx:onnx:FlattenDescription'));
            this.Type = getString(message('nnet_cnn_onnx:onnx:FlattenType'));
        end

        function Z = predict( this, X )
            % X is size [H W D C N].
            % Z is size [HWDC N].
            X = stripdims(X);
            [h,w,d,c,n] = size(X);
            Z = reshape(permute(X,[3 2 1 4 5]), [h*w*c*d n]);
            Z = dlarray(Z, 'CB');
        end
    end
end

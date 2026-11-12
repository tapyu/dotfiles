classdef FlattenInto2dLayer < nnet.layer.Layer & nnet.layer.Formattable ...
        & nnet.internal.cnn.layer.Traceable
    % FlattenInto2dLayer   Flatten layer ONNX-style into a 2D array
    %
    %   layer = FlattenInto2dLayer(Name) creates a layer that flattens a
    %   MATLAB 2D image batch in the way ONNX does, producing a 2D output
    %   array with CB format. The ONNX layer has input shape [N C H W] and
    %   output shape [N CHW], where the [C H W] dimensions have been
    %   flattened in that order in C-style. The corresponding MATLAB layer
    %   has input shape [H W C N] and will have an output shape of [CHW N].
    %   [C H W] C-style has the same linear ordering as [W H C]
    %   Fortran-style. So this layer creates [W H C N] in MATLAB, which has
    %   the desired linear ordering for each N. Then it reshapes to [CHW
    %   N].

    %   Copyright 2021 The MathWorks, Inc.

    methods
        function this = FlattenInto2dLayer(name)
            this.Name = name;
            this.Description = getString(message('nnet_cnn_onnx:onnx:FlattenDescription'));
            this.Type = getString(message('nnet_cnn_onnx:onnx:FlattenType'));
        end

        function Z = predict( ~, X )
            % X is size [H W C N].
            % Z is size [HWC N].
            X = stripdims(X);
            [h,w,c,n] = size(X);
            Z = reshape(permute(X,[2 1 3 4]), [h*w*c n]);
            Z = dlarray(Z, 'CB');
        end
    end

    methods (Static = true, Access = public, Hidden = true)       
         function name = matlabCodegenRedirect(~)
             name = 'nnet.internal.cnn.coder.onnx.FlattenInto2dLayer';
         end
     end 
end

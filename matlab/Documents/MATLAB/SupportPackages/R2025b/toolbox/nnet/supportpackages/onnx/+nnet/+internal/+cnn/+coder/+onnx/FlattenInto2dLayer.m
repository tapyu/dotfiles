classdef FlattenInto2dLayer < nnet.layer.Layer & nnet.layer.Formattable
    % Internal, codegen version of nnet.onnx.layer.FlattenInto2dLayer
    %#codegen

    %   Copyright 2021-2024 The MathWorks, Inc.

    methods
        function this = FlattenInto2dLayer(name)
            this.Name = name;
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

    methods (Static)
        function this_cg = matlabCodegenToRedirected(that)
            % Enable objects of class FlattenInto2dLayer to be passed from MATLAB 
            % to the generated MEX function
            this_cg = nnet.internal.cnn.coder.onnx.FlattenInto2dLayer(that.Name);
        end
         
        function this = matlabCodegenFromRedirected(this_cg)
            % Enable objects of class FlattenInto2dLayer to be successfully 
            % returned from the generated MEX function back to MATLAB
            this = nnet.onnx.layer.FlattenInto2dLayer(this_cg.Name);
        end

        function matlabCodegenValidateWithIOInfo(layer, validator, layerInfo)
            % Validate against unsupported formats
            unallowedFormat = 'U';
            dltargets.internal.validationUtils.errorIfInputContainsFormat(layer, validator, layerInfo, unallowedFormat);

            requiredFormat = 'C';
            dltargets.internal.validationUtils.errorIfInputDoesNotContainFormat(layer, validator, layerInfo, requiredFormat);
        end
    end
end

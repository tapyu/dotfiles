classdef CustomInputLayer < nnet.layer.Layer & nnet.internal.cnn.layer.Traceable
    % Layer that acts as the identity function. This layer is only used for
    % dlnetwork inputs whose data format is known, but for which a built-in
    % DLT input layer cannot be used. For example, for SSCB image tensors
    % with unknown height and/or width. The input X and output Z are both
    % in standard DLT dimension ordering.
    
    %   Copyright 2021-2024 The MathWorks, Inc.

    properties(SetAccess=protected)
        % A string informing the user about the dimension ordering they
        % must pass to this layer when predicting.
        InputInformation
        % The DLT tensor format string, e.g., "SSCB"
        DLTTensorFormat
    end
    
    methods
        function this = CustomInputLayer(name, inputInformation, dltTensorFormat)
            assert(isstring(name) || ischar(name), ...
                message('nnet_cnn_onnx:onnx:LayerNameArg'));
            assert(isstring(inputInformation) || ischar(inputInformation), message('nnet_cnn_onnx:onnx:CustomInputLayerBadInputInfo'));            
            this.Description = getString(message('nnet_cnn_onnx:onnx:CustomInputLayerDescription'));
            this.Type = getString(message('nnet_cnn_onnx:onnx:CustomInputLayerType', dltTensorFormat));
            this.InputInformation = inputInformation;            
            this.Name = name;
            if isempty(this.DLTTensorFormat)
                this.DLTTensorFormat = '';
            else
                this.DLTTensorFormat = dltTensorFormat;
            end 
        end
        
        function Z = predict( ~, X )
            Z = X;
        end
        
        function dLdX = backward( ~, ~, ~, dLdZ, ~ )
            dLdX = dLdZ;
        end
    end
    methods (Static = true, Access = public, Hidden = true)       
         function name = matlabCodegenRedirect(~)
             name = 'nnet.internal.cnn.coder.onnx.CustomInputLayer';
         end
    end
end

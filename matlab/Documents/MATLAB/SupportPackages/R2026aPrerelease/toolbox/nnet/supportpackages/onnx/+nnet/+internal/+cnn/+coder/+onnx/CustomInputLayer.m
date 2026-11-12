classdef CustomInputLayer < nnet.layer.Layer & nnet.internal.cnn.layer.Traceable
    % Internal, codegen version of nnet.onnx.layer.CustomInputLayer
    
    %   Copyright 2024 The MathWorks, Inc.
    %#codegen

    properties(SetAccess=protected)
        % A string informing the user about the dimension ordering they
        % must pass to this layer when predicting.
        InputInformation
        % The DLT tensor format string, e.g., "SSCB"
        DLTTensorFormat
    end
    
    methods
        function this = CustomInputLayer(name, ~, dltTensorFormat)
            this.Name = name;                   
            this.Description = getString(message('nnet_cnn_onnx:onnx:CustomInputLayerDescription'), matlab.internal.i18n.locale('en_US'));
            this.Type = getString(message('nnet_cnn_onnx:onnx:CustomInputLayerType', dltTensorFormat), matlab.internal.i18n.locale('en_US'));
            this.InputInformation = "";
            this.DLTTensorFormat = dltTensorFormat;
        end
        
        function Z = predict( ~, X )
            Z = X;
        end
    end

    methods (Static)
        function this_cg = matlabCodegenToRedirected(this)
            % Enable objects of class CustomInputLayer to be passed from MATLAB 
            % to the generated MEX function
            this_cg = nnet.internal.cnn.coder.onnx.CustomInputLayer(this.Name, this.InputInformation, this.DLTTensorFormat);
        end
         
        function this = matlabCodegenFromRedirected(this_cg)
            % Enable objects of class CustomInputLayer to be successfully 
            % returned from the generated MEX function back to MATLAB
            this = nnet.onnx.layer.CustomInputLayer(this_cg.Name, this_cg.InputInformation, this_cg.DLTTensorFormat);
        end
     end
end

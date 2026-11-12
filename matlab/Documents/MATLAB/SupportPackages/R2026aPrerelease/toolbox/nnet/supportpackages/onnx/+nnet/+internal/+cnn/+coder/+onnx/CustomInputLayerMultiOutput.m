classdef CustomInputLayerMultiOutput < nnet.layer.Layer & nnet.internal.cnn.layer.Traceable & nnet.layer.Formattable 
    % Internal, codegen version of nnet.onnx.layer.CustomInputLayerMultiOutput
    
    %   Copyright 2024-2026 The MathWorks, Inc.
    %#codegen
    
    properties(SetAccess=protected)
        NumDims             % An Integer value representing the number of dimensions 
                            % of the input data

        InputInformation    % A string informing the user about the dimension 
                            % ordering they must pass to this layer when predicting.    
        
        IsInputForwardONNX  % A logical flag for determining if the input needs to be permuted
                            % from forward to reverse ONNX dimension order

        TensorName          % Output tensor name

        DLTTensorFormat     % The DLT tensor format string, e.g., "SSCB"
    end

    methods(Static, Hidden)
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                'NumDims'
				'InputInformation'
                'IsInputForwardONNX'
				'TensorName'
				'DLTTensorFormat'
                };
        end
    end
    
    methods
        function this = CustomInputLayerMultiOutput(layerName, tensorName, numDims, ~, dltTensorFormat, isInputForwardONNX)
            this.Description = getString(message('nnet_cnn_onnx:onnx:CustomInputLayerDescription'), matlab.internal.i18n.locale('en_US'));
            this.Type = getString(message('nnet_cnn_onnx:onnx:CustomInputLayerType', dltTensorFormat), matlab.internal.i18n.locale('en_US'));
            this.InputInformation = "";            
            this.Name = layerName;
            this.NumDims = numDims;
            this.NumOutputs = 2;
            this.TensorName = tensorName;
            this.OutputNames = {tensorName, char(tensorName + "NumDims")};      
            this.IsInputForwardONNX = isInputForwardONNX;
            this.DLTTensorFormat = dltTensorFormat;
        end
        
        function [Z, numDims] = predict(this, X)
            if all(X.dims == 'U') && this.IsInputForwardONNX && this.NumDims > 1
                Z = permute(X, this.NumDims:-1:1);
            else
                Z = X;
            end
            numDims = dlarray(ones(1, this.NumDims, 'single'), 'UU');
        end
    end

    methods (Static)
        function this_cg = matlabCodegenToRedirected(this)
            % Enable objects of class CustomInputLayerMultiOutput to be passed from MATLAB 
            % to the generated MEX function
            this_cg = nnet.internal.cnn.coder.onnx.CustomInputLayerMultiOutput(this.Name, this.TensorName, this.NumDims, this.InputInformation, this.DLTTensorFormat, this.IsInputForwardONNX);
        end
         
        function this = matlabCodegenFromRedirected(this_cg)
            % Enable objects of class CustomInputLayerMultiOutput to be successfully 
            % returned from the generated MEX function back to MATLAB
            this = nnet.onnx.layer.CustomInputLayerMultiOutput(this_cg.Name, this_cg.TensorName, this_cg.NumDims, this_cg.InputInformation, this_cg.DLTTensorFormat, this_cg.IsInputForwardONNX);
        end
     end
end
classdef CustomOutputLayer < nnet.layer.Layer & nnet.layer.Formattable
    % Internal, codegen version of nnet.onnx.layer.CustomOutputLayer
    
    %   Copyright 2024-2026 The MathWorks, Inc.
    %#codegen
    properties(SetAccess=protected)
        OutputInformation   % A string informing the user about the dimension ordering of the
                            % output array that this layer will produce.

        DataFormat          % The dlarray format string
                
        IsInputForwardONNX  % A logical flag for determining if the input needs to be permuted
                            % from reverse to forward ONNX dimension order
    end

    methods(Static, Hidden)
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
				'OutputInformation'
                'DataFormat'
                'IsInputForwardONNX'
                };
        end
    end


    methods
        function this = CustomOutputLayer(name, ~, dltTensorFormat, isInputForwardONNX)
            this.Description = getString(message('nnet_cnn_onnx:onnx:CustomOutputLayerDescription'), matlab.internal.i18n.locale('en_US'));
            this.Type = getString(message('nnet_cnn_onnx:onnx:CustomOutputLayerType', dltTensorFormat), matlab.internal.i18n.locale('en_US'));            
            this.Name = name;                                   
            this.OutputInformation = "";             
            this.DataFormat = dltTensorFormat;
            this.IsInputForwardONNX = isInputForwardONNX;
        end

        function Z = predict(this, X)
            if all(char(this.DataFormat) == 'U') && this.IsInputForwardONNX
                X_ = permute(X, numel(char(this.DataFormat)):-1:1);
            else
                X_ = X;
            end
            Z = dlarray(X_, this.DataFormat);
        end
    end

    methods (Static)
        function this_cg = matlabCodegenToRedirected(this)
            % Enable objects of class CustomOutputLayer to be passed from MATLAB 
            % to the generated MEX function
            this_cg = nnet.internal.cnn.coder.onnx.CustomOutputLayer(this.Name, this.OutputInformation, this.DataFormat, this.IsInputForwardONNX);
        end
         
        function this = matlabCodegenFromRedirected(this_cg)
            % Enable objects of class CustomOutputLayer to be successfully 
            % returned from the generated MEX function back to MATLAB
            this = nnet.onnx.layer.CustomOutputLayer(this_cg.Name, this_cg.OutputInformation, this_cg.DataFormat, this_cg.IsInputForwardONNX);
        end
     end
end

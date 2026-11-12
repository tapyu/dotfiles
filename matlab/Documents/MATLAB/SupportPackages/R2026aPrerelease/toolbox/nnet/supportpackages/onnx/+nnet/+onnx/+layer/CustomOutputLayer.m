classdef CustomOutputLayer < nnet.layer.Layer & nnet.layer.Formattable
    % Layer implementing the Identity function. This is used in dlnetworks
    % and in layerGraphs with dlnetwork target, to designate the output
    % tensors, and to put them into the same ordering as in the onnx file.

    %   Copyright 2021-2024 The MathWorks, Inc.

    properties(SetAccess=protected)
        OutputInformation   % A string informing the user about the dimension ordering of the
                            % output array that this layer will produce.

        DataFormat          % The dlarray format string
                
        IsInputForwardONNX  % A logical flag for determining if the input needs to be permuted
                            % from reverse to forward ONNX dimension order
    end


    methods
        function this = CustomOutputLayer(name, outputInformation, dltTensorFormat, isInputForwardONNX)  
            assert(isstring(name) || ischar(name), ...
                message('nnet_cnn_onnx:onnx:LayerNameArg'));
            assert(isstring(outputInformation) || ischar(outputInformation), ...
                message('nnet_cnn_onnx:onnx:CustomOutputLayerBadOutputInfo'));                      
            this.Name = name;                   
            this.Description = getString(message('nnet_cnn_onnx:onnx:CustomOutputLayerDescription'));
            this.Type = getString(message('nnet_cnn_onnx:onnx:CustomOutputLayerType', dltTensorFormat));  
            this.OutputInformation = outputInformation; 
            this.DataFormat = dltTensorFormat;
            this.IsInputForwardONNX = isInputForwardONNX;
        end

        function Z = predict(this, X)
            if all(char(this.DataFormat) == 'U') && this.IsInputForwardONNX
                X = permute(X, numel(char(this.DataFormat)):-1:1);
            end
            Z = dlarray(X, this.DataFormat);
        end
    end
    methods (Static = true, Access = public, Hidden = true)       
         function name = matlabCodegenRedirect(~)
             name = 'nnet.internal.cnn.coder.onnx.CustomOutputLayer';
         end
    end
end

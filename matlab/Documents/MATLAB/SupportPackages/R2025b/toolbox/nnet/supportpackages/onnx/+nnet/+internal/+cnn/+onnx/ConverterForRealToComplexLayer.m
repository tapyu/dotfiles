classdef ConverterForRealToComplexLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.cnn.layer.RealToComplexLayer into ONNX
    
    % Copyright 2024 The MathWorks, Inc.
    
    methods
        function this = ConverterForRealToComplexLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, ~, ~, ~)
            error(message("nnet_cnn_onnx:onnx:CantExportLayer", class(this.NNTLayer)));
        end
    end
end

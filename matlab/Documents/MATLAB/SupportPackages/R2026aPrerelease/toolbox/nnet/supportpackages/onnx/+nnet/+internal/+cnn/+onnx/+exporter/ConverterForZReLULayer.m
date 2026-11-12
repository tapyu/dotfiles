classdef ConverterForZReLULayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a nnet.cnn.layer.ZReLULayer into ONNX

    % Copyright 2025 The MathWorks, Inc.

    methods
        function this = ConverterForZReLULayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, ~, ~, ~)
            error(message("nnet_cnn_onnx:onnx:CantExportLayer", class(this.NNTLayer)));
        end
    end
end
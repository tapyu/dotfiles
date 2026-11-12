classdef ConverterForSpectralConvolution1DLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a nnet.cnn.layer.SpectralConvolution1DLayer into ONNX

    % Copyright 2025 The MathWorks, Inc.

    methods
        function this = ConverterForSpectralConvolution1DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, ~, ~, ~) %#ok<STOUT>
            error(message("nnet_cnn_onnx:onnx:CantExportLayer", class(this.NNTLayer)));
        end
    end
end
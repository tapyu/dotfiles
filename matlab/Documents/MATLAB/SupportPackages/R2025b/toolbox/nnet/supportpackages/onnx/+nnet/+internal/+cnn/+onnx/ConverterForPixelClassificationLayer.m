classdef ConverterForPixelClassificationLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a PixelClassificationLayer into ONNX
        
    % Copyright 2018-2021 The MathWorks, Inc.

    methods
        function this = ConverterForPixelClassificationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            inputTensorNames        = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            dltOutputSize           = this.InputLayerSizes{1};
            assert(numel(dltOutputSize)==3 || numel(dltOutputSize)==4,...
                message('nnet_cnn_onnx:onnx:UnexpectedDLTInputSize', this.NNTLayer.Name, numel(dltOutputSize)+1));
            outputTensorSize        = [{this.BatchSizeToExport}, num2cell(dltOutputSize([end 1:(end-1)]))];	% {N C H W} or {N C H W D}

            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = makeValueInfoProtoFromDimensions(inputTensorNames{1}, TensorProto_DataType.FLOAT, outputTensorSize);                                                 
        end
    end
end

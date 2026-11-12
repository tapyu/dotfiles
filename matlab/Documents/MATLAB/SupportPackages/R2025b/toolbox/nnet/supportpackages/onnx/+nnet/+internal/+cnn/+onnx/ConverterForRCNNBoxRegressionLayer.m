classdef ConverterForRCNNBoxRegressionLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a RCNNBoxRegressionLayer into ONNX
    
    % Copyright 2018-2021 The MathWorks, Inc.
    
    properties
        LayerAnalyzer
    end
    
    methods
        function this = ConverterForRCNNBoxRegressionLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
            this.LayerAnalyzer = layerAnalyzer;
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            inputLayerName          = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            parameterInitializers   = [];
            networkInputs           = [];
            hwcSize                 = this.LayerAnalyzer.Inputs.Size{1}(:)';
            networkOutputs          = makeValueInfoProtoFromDimensions(...
                inputLayerName{1}, ...
                TensorProto_DataType.FLOAT, ...
                num2cell([1 hwcSize([3 1 2])]));	% NNT output size is hwcn, ONNX is nchw with n=1
        end
    end
end

classdef ConverterForYOLOv2ReorgLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a yolov2ReorgLayer into ONNX.

    % Copyright 2019-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForYOLOv2ReorgLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
            [ONNXName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            ONNXName                = makeUniqueName({nodeProto.name}, ONNXName);
            [SpaceToDepthNode, inits] = createNodeProto(this, 'SpaceToDepth', ONNXName, inputTensorNames, {ONNXName}, this.NNTLayer.Stride(1));          

            nodeProto(end+1)        = SpaceToDepthNode;
            parameterInitializers   = inits;
            networkInputs           = [];
            networkOutputs          = [];
        end
    end
end

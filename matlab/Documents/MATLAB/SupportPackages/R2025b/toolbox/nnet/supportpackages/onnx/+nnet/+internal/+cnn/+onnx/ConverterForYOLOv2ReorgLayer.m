classdef ConverterForYOLOv2ReorgLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a yolov2ReorgLayer into ONNX.

    % Copyright 2019-2021 The MathWorks, Inc.
    
    methods
        function this = ConverterForYOLOv2ReorgLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
            
            [ONNXName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            ONNXName                = makeUniqueName({nodeProto.name}, ONNXName);
            newNode                 = NodeProto;
            newNode.op_type         = 'SpaceToDepth';
            newNode.name            = ONNXName;
            newNode.input           = inputTensorNames;
            newNode.output{1}       = ONNXName;
            newNode.attribute       = makeAttributeProto('blocksize', 'INT', this.NNTLayer.Stride(1));
            
            nodeProto(end+1)        = newNode;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];
        end
    end
end

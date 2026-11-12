classdef ConverterForAdditionLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an additionLayer into ONNX

    % Copyright 2018-2024 The MathWorks, Inc.

    methods
        function this = ConverterForAdditionLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            inputTensorNames    = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});

            [ONNXName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            ONNXName          = makeUniqueName({nodeProto.name}, ONNXName);
            newNode           = NodeProto;
            newNode.op_type   = 'Sum';
            newNode.name      = ONNXName;
            newNode.input     = inputTensorNames;
            newNode.output{1} = ONNXName;

            nodeProto(end+1)        = newNode;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            TensorNameMap(this.NNTLayer.Name) = ONNXName;
            outputTensorLayout                = inputTensorLayout;
            outputTensorName                  = ONNXName;
            TensorLayoutMap(outputTensorName) = outputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

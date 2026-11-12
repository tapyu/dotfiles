classdef ConverterForTanhLayers < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert the following layers into ONNX Tanh operator:
    % nnet.cnn.layer.TanhLayer
    % nnet.onnx.layer.TanhLayer
    % nnet.keras.layer.TanhLayer

    % Copyright 2020-2024 The Mathworks, Inc.

    methods
        function this = ConverterForTanhLayers(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [NodeName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            NodeName                = makeUniqueName({nodeProto.name}, NodeName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
            newNode                 = NodeProto;
            newNode.op_type         = 'Tanh';
            newNode.name            = NodeName;
            newNode.input           = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            newNode.output          = {NodeName};

            nodeProto(end+1)        = newNode;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            outputTensorName = NodeName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end


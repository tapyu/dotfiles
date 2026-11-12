classdef ConverterForONNXSigmoidLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.onnx.layer.SigmoidLayer into ONNX

    % Copyright 2018-2024 The MathWorks, Inc.

    methods
        function this = ConverterForONNXSigmoidLayer(layerAnalyzer)
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
            newNode.op_type         = 'Sigmoid';
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


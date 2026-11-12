classdef ConverterForLeakyReLULayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a LeakyReluLayer into ONNX

    % Copyright 2018-2024 The MathWorks, Inc.

    methods
        function this = ConverterForLeakyReLULayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            % Make the nodeProto
            newNode           = NodeProto;
            newNode.op_type	= 'LeakyRelu';
            newNode.name      = onnxName;
            newNode.input     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            newNode.output    = {onnxName};
            newNode.attribute = makeAttributeProto('alpha', 'FLOAT', this.NNTLayer.Scale);

            nodeProto(end+1)        = newNode;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            outputTensorName = onnxName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end


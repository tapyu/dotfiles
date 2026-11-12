classdef ConverterForDropoutLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a DropoutLayer into ONNX

    % Copyright 2018-2024 The MathWorks, Inc.

    methods
        function this = ConverterForDropoutLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});

            % Make the nodeProto
            dropoutInput        = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            dropoutOutput       = {onnxName};
            [dropoutNode, init] = createNodeProto(this, 'Dropout', onnxName, dropoutInput, dropoutOutput, this.NNTLayer.Probability);

            nodeProto(end+1)        = dropoutNode;
            parameterInitializers   = init;
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            TensorNameMap(this.NNTLayer.Name) = onnxName;
            TensorLayoutMap(onnxName) = inputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end


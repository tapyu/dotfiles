classdef ConverterForSigmoidLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a SigmoidLayer into ONNX

    % Copyright 2020-2025 The MathWorks, Inc.

    methods
        function this = ConverterForSigmoidLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers,...
                networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap]...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});

            % Make the nodeProto
            [newNode, ~] = createNodeProto(this, 'SISO', 'Sigmoid', onnxName, ...
                        mapTensorNames(this, this.InputLayerNames(1), TensorNameMap), {onnxName});
            nodeProto(end+1)        = newNode;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];

            TensorNameMap(this.NNTLayer.Name) = onnxName;
            TensorLayoutMap(onnxName) = inputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

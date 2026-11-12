classdef ConverterForCrossChannelNormalizationLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a CrossChannelNormalizationLayer into ONNX

    % Copyright 2018-2025 The MathWorks, Inc.

    methods
        function this = ConverterForCrossChannelNormalizationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames            = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout           = TensorLayoutMap(inputTensorNames{1});
            [newNode, ~] = createNodeProto(this, 'LRN', onnxName, ...
                mapTensorNames(this, this.InputLayerNames(1), TensorNameMap), {onnxName},...
                this.NNTLayer.Alpha, this.NNTLayer.Beta, this.NNTLayer.K, this.NNTLayer.WindowChannelSize);

            nodeProto(end+1)        = newNode;
            parameterInitializers   = [];
            networkInputs       	= [];
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

classdef ConverterForDepthConcatenationLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a DepthConcatenationLayer into ONNX

    % Copyright 2018-2025 The MathWorks, Inc.

    methods
        function this = ConverterForDepthConcatenationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % nodeProto is a NodeProto.
            % parameterInitializers is a TensorProto array.
            % parameterInputs is a ValueInfoProto array describing the parameters of the node.
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
            % All our defined ONNX tensor layouts contain a 'c' character.
            % Locate the 'c' and set the ONNX axis to that location
            % (origin-0)
            idx = strfind(inputTensorLayout, 'c');
            assert(~isempty(idx))
            concatAxis = idx-1;
            [newNode, ~] = createNodeProto(this, 'Concat', onnxName,...
                mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap),...
                {onnxName},concatAxis);

            nodeProto(end+1)        = newNode;
            parameterInitializers   = [];
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

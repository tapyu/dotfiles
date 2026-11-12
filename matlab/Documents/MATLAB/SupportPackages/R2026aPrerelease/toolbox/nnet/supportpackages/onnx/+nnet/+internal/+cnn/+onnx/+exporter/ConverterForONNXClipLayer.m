classdef ConverterForONNXClipLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a nnet.onnx.layer.ClipLayer into ONNX

    % Copyright 2018-2025 The MathWorks, Inc.

    methods
        function this = ConverterForONNXClipLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % Since this layer was derived from ONNX's Clip, we can just
            % generate a Clip.
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});

            clipName                = makeUniqueName({nodeProto.name}, onnxName);
            clipInput               = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            clipOutput              = {clipName};
            clipMin                 = [];
            clipMax                 = [];

            if this.NNTLayer.Min > -Inf
                clipMin = this.NNTLayer.Min;
            end
            if this.NNTLayer.Max < Inf
                clipMax = this.NNTLayer.Max;
            end
            [clipNode, inits] = createNodeProto(this, 'Clip', clipName, clipInput, clipOutput, clipMin, clipMax);

            nodeProto(end+1)        = clipNode;
            parameterInitializers   = inits;
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            outputTensorName = clipName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
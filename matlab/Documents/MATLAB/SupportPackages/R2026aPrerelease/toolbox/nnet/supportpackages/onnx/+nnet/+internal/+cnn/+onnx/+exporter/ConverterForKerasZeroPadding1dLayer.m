classdef ConverterForKerasZeroPadding1dLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a nnet.keras.layer.ZeroPadding1dLayer into ONNX

    % Copyright 2021-2025 The MathWorks, Inc.

    methods
        function this = ConverterForKerasZeroPadding1dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});

            padName     = onnxName;
            padInput    = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            padOutput   = {onnxName};

            % Assuming the input tensor is [S N C]
            pads                = [this.NNTLayer.leftPad, 0, 0, this.NNTLayer.rightPad, 0, 0];
            value               = 0;
            mode                = 'constant';
            [padNode, padInits] = createNodeProto(this, 'Pad', padName, padInput, padOutput, pads, mode, value);

            nodeProto(end+1)        = padNode;
            parameterInitializers   = padInits;
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            outputTensorName            = onnxName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

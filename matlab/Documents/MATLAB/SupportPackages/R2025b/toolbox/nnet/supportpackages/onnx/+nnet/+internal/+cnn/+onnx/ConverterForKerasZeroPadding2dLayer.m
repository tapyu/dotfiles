classdef ConverterForKerasZeroPadding2dLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.keras.layer.ZeroPadding2dLayer into ONNX

    % Copyright 2018-2024 The MathWorks, Inc.

    methods
        function this = ConverterForKerasZeroPadding2dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});

            padName      = onnxName;
            padInput     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            padOutput    = {onnxName};

            % Assuming the input tensor is [N C H W]
            pads                = [0, 0, this.NNTLayer.Top, this.NNTLayer.Left, 0, 0, this.NNTLayer.Bottom, this.NNTLayer.Right];
            mode                = 'constant';
            value               = 0;
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

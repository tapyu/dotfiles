classdef ConverterForNASNetMobileZeroPadding2dLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.nasnetmobile.layer.NASNetMobileZeroPadding2dLayer into ONNX

    % Copyright 2019-2024 The MathWorks, Inc.

    methods
        function this = ConverterForNASNetMobileZeroPadding2dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
            outputTensorLayout      = inputTensorLayout;

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

            % Make maps
            TensorNameMap(this.NNTLayer.Name) = onnxName;
            TensorLayoutMap(onnxName) = outputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

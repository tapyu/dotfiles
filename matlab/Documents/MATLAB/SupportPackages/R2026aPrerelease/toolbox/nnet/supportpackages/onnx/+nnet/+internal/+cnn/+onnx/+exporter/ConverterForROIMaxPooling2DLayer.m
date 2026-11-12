classdef ConverterForROIMaxPooling2DLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a roiMaxPooling2DLayer into ONNX

    % Copyright 2018-2025 The MathWorks, Inc.

    methods
        function this = ConverterForROIMaxPooling2DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});

            OutputSize      = this.NNTLayer.OutputSize;
            internalLayer   = iGetInternalLayer(this.NNTLayer);
            ScaleFactor     = internalLayer.ScaleFactor;
            if ScaleFactor(1) ~= ScaleFactor(2)
                warning(message('nnet_cnn_onnx:onnx:ROIMaxPooling2DLayerScalesUnequal', ...
                    this.NNTLayer.Name, ScaleFactor(1), ScaleFactor(2), ScaleFactor(1)));
            end
            ScaleFactor = ScaleFactor(1);
            [newNode, ~] = createNodeProto(this, 'MaxRoiPool', onnxName, ...
                        mapTensorNames(this, this.InputLayerNames, TensorNameMap), {onnxName},...
                        OutputSize, ScaleFactor);
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

function L = iGetInternalLayer(ExtLayer)
C = nnet.cnn.layer.Layer.getInternalLayers(ExtLayer);
L = C{1};
end

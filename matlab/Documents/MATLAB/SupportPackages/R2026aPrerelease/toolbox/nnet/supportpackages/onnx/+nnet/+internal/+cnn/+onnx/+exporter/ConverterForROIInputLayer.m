classdef ConverterForROIInputLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a ROIInputLayer into ONNX

    % Copyright 2018-2025 The MathWorks, Inc.

    methods
        function this = ConverterForROIInputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            % Generate a ValueInfoProto describing the input tensor.
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            outputSize              = this.OutputSize{1};

            inputTensorSize = num2cell([outputSize(1), 5]); % NNT is size [numROI 4], ONNX is size [numROI 5]
            networkInputs  	= makeValueInfoProtoFromDimensions(onnxName, TensorProto_DataType.FLOAT, inputTensorSize);
            networkOutputs	= [];
            parameterInitializers = [];

            % Update maps
            TensorNameMap(this.NNTLayer.Name) = onnxName;
            TensorLayoutMap(onnxName) = 'nc';             % It's really [numROI,5]

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

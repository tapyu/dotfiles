classdef ConverterForMultiplicationLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an multiplicationLayer into ONNX

    % Copyright 2020-2024 The MathWorks, Inc.

    methods
        function this = ConverterForMultiplicationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            inputTensorNames    = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            layerName = this.NNTLayer.Name;

            %Since multiplication layer in ONNX supports only two inputs, we are creating
            %(n-1) instances of multiplication layer for n input multiplication layer.
            for i=2:numel(inputTensorNames)
                newNode           = NodeProto;
                newNode.op_type   = 'Mul';

                % name of the new node
                if i==numel(inputTensorNames)
                    [ONNXName, nameChanged] = legalizeNNTName(this,layerName);
                    ONNXName = makeUniqueName({nodeProto.name}, ONNXName);
                    newNode.name = ONNXName;
                else
                    ONNXName = [layerName,'_',num2str(i-1)];
                    [ONNXName, nameChanged] = legalizeNNTName(this,ONNXName);
                    ONNXName = makeUniqueName({nodeProto.name}, ONNXName);
                    newNode.name = ONNXName;
                end

                % input to the new node
                if i==2
                    newNode.input = inputTensorNames(1:i);
                else
                    newNode.input = mapTensorNames(this, {[layerName,'_',num2str(i-2)] ,inputTensorNames{i}}, TensorNameMap);
                end

                % output of the new node
                newNode.output{1} = ONNXName;

                % addition new node to nodeProto
                nodeProto(end+1)        = newNode;
                parameterInitializers   = [];
                networkInputs           = [];
                networkOutputs          = [];
            end
            % Update maps
            TensorNameMap(this.NNTLayer.Name) = ONNXName;
            outputTensorName = ONNXName;
            outputTensorLayout = inputTensorLayout;
            TensorLayoutMap(outputTensorName) = outputTensorLayout;

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
